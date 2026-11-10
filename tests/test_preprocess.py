"""Physical raw-input preparation, resume identity and supervision separation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from eatdet.preprocess import discover_people, prepare, _save_imu_frames, _prepare_native, load_prepared, DEVICE_SETTINGS


class PreprocessingTests(unittest.TestCase):
    def test_official_tables_preserve_ids_times_and_hand_relations(self):
        from eatdet.annotations import convert_tables
        import csv, io
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);primary=root/'meals.csv';secondary=root/'secondary.csv';users=root/'users.csv'
            fields=['uniqueid','healthid','externalid','timeFrame','beforeTime','afterTime','wearHand','dietaryHand']
            rows=[dict(uniqueid='M1',healthid='H',externalid='P',timeFrame="{'startTime': 1700000000000, 'endTime': 1700000300000}",
                beforeTime='invalid unused display time',afterTime='',wearHand='左手',dietaryHand='右手'),
                dict(uniqueid='BAD',healthid='H',externalid='PJ',timeFrame="{'startTime': 1700000000000, 'endTime': 1700000300000}")]
            with primary.open('w',encoding='gb18030',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            secondary.write_text(','.join(fields)+'\n'+ '"unfinished',encoding='utf-8')
            users.write_text('healthid,externalid\nH,P\n',encoding='utf-8')
            result=convert_tables([primary,secondary],root/'annotations',users_csv=users,people=['P'])
            self.assertEqual(result['events'],1)
            with (root/'annotations/events.csv').open(encoding='utf-8') as stream:parsed=list(csv.DictReader(stream))
            self.assertEqual(parsed[0]['event_id'],'M1');self.assertEqual(parsed[0]['beforeTime'],'1700000000000')
            metadata=json.loads((root/'annotations/event_metadata.jsonl').read_text())
            self.assertFalse(metadata['same_hand'])
            self.assertEqual(json.loads((root/'annotations/conversion.json').read_text())['skipped'][0]['event_id'],'BAD')

    def test_discovery_normalizes_externalid_and_rejects_missing_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/' hnu21001';folder.mkdir();(folder/'record.txt').write_text('placeholder')
            self.assertEqual(list(discover_people(root)),['HNU21001'])
            with self.assertRaises(ValueError):discover_people(root,['MISSING'])
            other=root/'HNU21001';other.mkdir();(other/'another.txt').write_text('placeholder')
            with self.assertRaises(ValueError):discover_people(root)

    def test_work_directory_cannot_modify_original_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):prepare(root,root/'cache')
            with self.assertRaises(ValueError):prepare(root,root)

    def test_multiview_frames_do_not_depend_on_annotations(self):
        from signals.competition.annotations import CompetitionAnnotations
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);device=root/'device';device.mkdir();source=device/'record';source.mkdir()
            start=25000000000;n=900;ticks=start+np.arange(n)
            t=np.arange(n)/15;x=np.column_stack([np.sin(t),np.cos(t),np.ones(n),np.sin(t/2),np.cos(t/2),np.zeros(n)]).astype(np.float32)
            arrays=dict(data=x,ticks=ticks,observed_mask=np.ones((n,2),bool),imu_support=np.ones(n,bool),continuous_segment=np.zeros(n,np.int32))
            for name,value in arrays.items():np.save(source/(name+'.npy'),value)
            entry=dict(participant_key='P',start_ms=int(start*1000/15),key='P/record/0',recording_key='P/record',path='record')
            (device/'manifest.json').write_text(json.dumps(dict(sessions=[entry])))
            labels=CompetitionAnnotations();split=dict(train=['P'],val=[],test=[])
            first=_save_imu_frames(device,root/'one',labels,{'P'},split)
            ann=root/'annotations';ann.mkdir();begin=int(ticks[100]*1000/15);end=int(ticks[600]*1000/15)
            (ann/'events.csv').write_text(f'externalid,beforeTime,afterTime\nP,{begin:013d},{end:013d}\n')
            (ann/'policy.json').write_text(json.dumps(dict(coverage='all_observed')))
            second=_save_imu_frames(device,root/'two',CompetitionAnnotations(ann),{'P'},split)
            for name in ('imu','point_mask','frame_quality','segment_ids'):
                a=np.load(root/'one'/first['records'][0]['path']/(name+'.npy'))
                b=np.load(root/'two'/second['records'][0]['path']/(name+'.npy'))
                np.testing.assert_array_equal(a,b)
            self.assertFalse(np.load(root/'one'/first['records'][0]['path']/'label_mask.npy').any())
            self.assertTrue(np.load(root/'two'/second['records'][0]['path']/'label_mask.npy').any())

    def test_prepared_paths_relocate_and_missing_artifacts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'imu').mkdir();(root/'imu/manifest.json').write_text('{}');(root/'native/ppg').mkdir(parents=True)
            value=dict(format='eating-detector.prepared.1',complete=True,imu_manifest='imu/manifest.json',native_root='native',ppg_cache_root='native/ppg')
            (root/'prepared.json').write_text(json.dumps(value))
            result=load_prepared(root)
            self.assertEqual(result['imu_manifest'],str(root/'imu/manifest.json'))
            (root/'imu/manifest.json').unlink()
            with self.assertRaises(FileNotFoundError):load_prepared(root)

    def test_hardware_scales_and_clock_contract_are_fixed(self):
        self.assertEqual(DEVICE_SETTINGS['acc_scale_g'],.00024414)
        self.assertAlmostEqual(DEVICE_SETTINGS['gyro_scale_dps'],.06999997270451806)
        self.assertEqual(DEVICE_SETTINGS['session_gap_ms'],3000)
        self.assertEqual(DEVICE_SETTINGS['max_interpolation_gap_ms'],250)


if __name__=='__main__':unittest.main()
