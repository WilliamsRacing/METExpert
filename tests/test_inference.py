"""Numerical parity, label independence and missing-modality prediction checks."""
from pathlib import Path
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from eatdet.inference import _SignalWindows, infer_imu_person, infer_ppg_person, _join_ppg, _native_session, predict_from_sessions, write_predictions, verify_weights, COMPONENT_FILES


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, root):
        rng = np.random.default_rng(23)
        folder = Path(root) / "signal"
        folder.mkdir()
        n = 12
        arrays = dict(imu=rng.normal(size=(n, 12, 450)).astype(np.float32),
            point_mask=np.ones((n, 450), bool), frame_quality=np.ones((n, 8), np.float32),
            segment_ids=np.zeros(n, np.int64), target=np.zeros(n, np.int64),
            target_fraction=np.zeros(n, np.float32), boundary=np.zeros((n, 2), np.float32),
            label_mask=np.ones(n, bool))
        arrays["point_mask"][4] = False
        arrays["point_mask"][2, :20] = False
        arrays["segment_ids"][7:] = 1
        for name, value in arrays.items():
            np.save(folder / (name + ".npy"), value)
        return dict(root=str(root), split=dict(train=["PERSON"], val=[]),
            normalization=dict(mean=[.1] * 12, std=[1.2] * 12, people=["PERSON"]),
            records=[dict(person="PERSON", path="signal", frame_count=n, start_tick=45000,
                source_key="signal", supported_points=int(arrays["point_mask"].sum()),
                positive_frames=[], labeled_frames=list(range(n)))])

    def test_prediction_windows_never_open_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(tmp)
            window = _SignalWindows(manifest, "PERSON", manifest["normalization"])
            real_load = np.load
            def only_signal(path, *args, **kwargs):
                self.assertIn(Path(path).name, ("imu.npy", "point_mask.npy", "frame_quality.npy", "segment_ids.npy"))
                return real_load(path, *args, **kwargs)
            with patch("eatdet.inference.np.load", side_effect=only_signal):
                out = window.window(0, -3, 20)
            self.assertEqual(out["imu"].shape, (20, 12, 450))
            self.assertFalse(out["point_mask"][:3].any())
            self.assertFalse(out["point_mask"][-5:].any())
            window.close()

    def test_primary_component_matches_reference_numerical_forward(self):
        from motion_encoder.model import IMUV6Model
        from primary_fusion.prepare import infer_person
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(tmp)
            torch.manual_seed(37)
            model = IMUV6Model()
            ck = dict(state_dict=model.state_dict(), model_config=model.config.to_dict(),
                      normalization=manifest["normalization"])
            got = infer_imu_person(manifest, ck, "PERSON", batch_size=2)
            reference, _ = infer_person(dict(imu_project=tmp, device="cpu", amp=False),
                manifest, ck, "train", "PERSON", {"signal": 0})
            for name in reference:
                np.testing.assert_allclose(got[name], reference[name], atol=1e-6, rtol=1e-6, err_msg=name)

    def test_context_component_matches_reference_numerical_forward(self):
        from context_encoder.model import IMUV8Model
        from context_encoder.train import predict_records
        from event_detection.prepare import IMUAccumulator
        with tempfile.TemporaryDirectory() as tmp:
            manifest = self.fixture(tmp)
            torch.manual_seed(41)
            model = IMUV8Model()
            ck = dict(state_dict=model.state_dict(), model_config=model.config.to_dict(),
                      normalization=manifest["normalization"])
            got = infer_imu_person(manifest, ck, "PERSON", contextual=True, batch_size=2)
            rows = predict_records(model, manifest, ["PERSON"], ck["normalization"], batch_size=2)
            acc = IMUAccumulator(np.concatenate([row["frame_start_tick"] for row in rows]))
            for i, row in enumerate(rows):
                acc.add(row, i)
            reference = acc.finish()
            for name in reference:
                np.testing.assert_allclose(got[name], reference[name], atol=1e-6, rtol=1e-6, err_msg=name)

    def test_ppg_alignment_matches_file_reference(self):
        from event_detection.prepare import join_ppg
        with tempfile.TemporaryDirectory() as tmp:
            starts = np.array([0, 450, 900, 1350], np.int64)
            arrays = dict(frame_start_tick=starts[1:3], ppg_logits=np.array([1., 2.], np.float32),
                ppg_start_logits=np.array([-.2, .1], np.float32), ppg_end_logits=np.array([-.1, .2], np.float32),
                ppg_quality=np.array([.2, .8], np.float32), ppg_quality_raw_max=np.array([.4, .9], np.float32),
                ppg_quality_expert=np.array([.2, .8], np.float32), wave_coverage=np.ones(2, np.float32),
                feature_coverage=np.ones(2, np.float32), raw_coverage=np.ones(2, np.float32),
                spectral_coverage=np.ones(2, np.float32), ppg_embedding=np.ones((2, 64), np.float32),
                ppg_support=np.ones(2, bool))
            np.savez(Path(tmp) / "record.npz", **arrays)
            expected, expected_ids = join_ppg(starts, [dict(path="record.npz")], tmp)
            got, ids = _join_ppg(starts, [dict(arrays=arrays)])
            self.assertEqual(set(got), set(expected))
            for name in expected:
                np.testing.assert_array_equal(got[name], expected[name])
            np.testing.assert_array_equal(ids, expected_ids)

    def test_native_physical_clock_does_not_read_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "session"
            folder.mkdir()
            ticks = np.arange(45123, 46001, dtype=np.int64)
            np.save(folder / "imu__ticks.npy", ticks)
            np.save(folder / "target__observed.npy", np.ones(len(ticks), bool))
            got = _native_session(tmp, dict(path="session"), include_labels=False)
            self.assertEqual(set(got), {"ticks", "physical_mask"})
            self.assertEqual(int(got["physical_mask"].sum()), len(ticks))
            self.assertEqual(got["ticks"][0, 0], 45000)

    def test_empty_ppg_and_empty_imu_are_usable_missing_modalities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "records.jsonl").write_text("")
            self.assertEqual(infer_ppg_person(root, {}, "NEW_PERSON"), [])
            from motion_encoder.model import IMUV6Model
            model = IMUV6Model()
            ck = dict(state_dict=model.state_dict(), model_config=model.config.to_dict(),
                      normalization=dict(mean=[0.] * 12, std=[1.] * 12))
            got = infer_imu_person(dict(root=tmp, records=[]), ck, "NEW_PERSON")
            self.assertEqual(got["imu_logits"].shape, (0, 450))

    def test_event_path_preserves_primary_and_official_csv_schema(self):
        primary = [dict(participant_key="NEW", start_ms=1760000001000, end_ms=1760000002000, score=.7)]
        rescue = [dict(participant_key="NEW", session_key="S", sources=["imu"], start_ms=1760000003000, end_ms=1760000004000)]
        payload = dict(candidate_rule={}, threshold=.5, nms_iou=.5, nms_ios=.8)
        import event_detection.ranker  # Resolve the original generator before temporary mocks.
        with patch("eatdet.inference.primary_events", return_value=primary), \
             patch('event_detection.events.generate_candidates', return_value=rescue), \
             patch('event_detection.ranker.predict_ranker', return_value=np.array([.8])):
            result = predict_from_sessions([{}], payload, {})
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], primary[0])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_predictions(result, Path(tmp) / "predictions.csv")
            self.assertEqual(path.read_text().splitlines(), ["externalid,beforeTime,afterTime", "NEW,1760000001000,1760000002000", "NEW,1760000003000,1760000004000"])

    def test_invalid_csv_preserves_existing_output_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            previous = b"externalid,beforeTime,afterTime\r\nOLD,1760000001000,1760000002000\r\n"
            path.write_bytes(previous)
            invalid = [dict(participant_key="NEW", start_ms=1760000003000, end_ms=1760000004000),
                       dict(participant_key="NEW", start_ms=1000, end_ms=2000)]
            with self.assertRaises(ValueError):write_predictions(invalid, path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            invalid[1] = dict(participant_key="NEW", start_ms=1760000001000.0, end_ms=1760000002000.0)
            with self.assertRaises(ValueError):write_predictions(invalid, path)
            self.assertEqual(path.read_bytes(), previous)
            self.assertEqual(list(Path(tmp).iterdir()), [path])
            write_predictions([], path)
            self.assertEqual(path.read_text().splitlines(), ["externalid,beforeTime,afterTime"])

    def test_weight_manifest_detects_changes_traversal_and_nonfinite_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "weights"
            root.mkdir()
            components = {}
            for name, filename in COMPONENT_FILES.items():
                data = name.encode()
                (root / filename).write_bytes(data)
                components[name] = dict(path="weights/" + filename, sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
            manifest = dict(format="eating-detection.weights.1", components=components)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            self.assertTrue(verify_weights(root)["verified"])
            (root / "ppg.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):verify_weights(root)
            (root / "ppg.pt").write_bytes(b"ppg")
            manifest["components"]["ppg"]["path"] = "../ppg.pt"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_weights(root)
            manifest["components"]["ppg"]["path"] = "weights/ppg.pt"
            manifest["decoder"] = dict(threshold=float("nan"))
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):verify_weights(root)


if __name__ == "__main__":
    unittest.main()
