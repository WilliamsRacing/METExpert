"""Training ownership, optimizer and deployment checkpoint integration checks."""
from pathlib import Path
import copy
import json
import tempfile
import unittest

import numpy as np
import torch

from eatdet.training import load_training_inputs, train_primary_imu, train_primary_fusion, FusionEvidenceDataset, train_event_ranker


class TrainingTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_split_leakage_is_rejected_before_signal_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);split=root/'split.json'
            prepared=dict(imu_manifest='NEVER_OPEN',native_root='NEVER_OPEN',ppg_cache_root='NEVER_OPEN',
                annotation_root=str(root),split_file=str(split))
            for bad in (dict(train=['A'],val=['A'],test=[]),dict(train=['A','B'],val=['C'],test=['A']),
                        dict(train=['A'],val=['B'],test=[],shared_source_atomic_groups=[['A','B']])):
                split.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):load_training_inputs(prepared)
            split.write_text(json.dumps(dict(train=['A'],val=['B'],test=['C'])))
            with self.assertRaises(FileNotFoundError):load_training_inputs(prepared)

    def test_primary_imu_real_optimizer_exports_prediction_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rng=np.random.default_rng(17);n=24;records=[]
            for person in ('A','B'):
                folder=root/person;folder.mkdir();target=np.zeros(n,np.int64);target[6:14]=1
                arrays=dict(imu=rng.normal(size=(n,12,450)).astype(np.float32),point_mask=np.ones((n,450),bool),
                    frame_quality=np.ones((n,8),np.float32),segment_ids=np.zeros(n,np.int64),target=target,
                    target_fraction=(target>0).astype(np.float32),boundary=np.zeros((n,2),np.float32),label_mask=np.ones(n,bool))
                for key,value in arrays.items():np.save(folder/(key+'.npy'),value)
                records.append(dict(person=person,path=person,start_tick=45000,frame_count=n,supported_points=n*450,
                    positive_frames=list(range(6,14)),labeled_frames=list(range(n)),source_key=person))
            manifest=root/'manifest.json';manifest.write_text(json.dumps(dict(records=records,split=dict(train=['A'],val=['B'],test=[]))))
            config=dict(seed=17,epochs=1,schedule_epochs=30,samples_per_epoch=4,batch_size=2,workers=0,train_frames=48,
                core_frames=16,lr=.0003,weight_decay=.001,normalization_points_per_person=32,
                model=dict(local_channels=8,embedding_dim=16,dropout=0),loss=dict(role_weight=.2,dice_weight=.2,boundary_weight=.1))
            history=train_primary_imu(dict(imu_manifest=str(manifest)),dict(train=['A'],val=['B'],test=[]),config,
                root/'model.pt',torch.device('cpu'),False,smoke=True)
            self.assertEqual(len(history),1);self.assertTrue(np.isfinite(history[0]['loss']))
            ck=torch.load(root/'model.pt',weights_only=True);self.assertEqual(ck['normalization']['people'],['A'])
            from motion_encoder.model import IMUV6Model, ModelConfig
            model=IMUV6Model(ModelConfig(**ck['model_config']));model.load_state_dict(ck['state_dict'],strict=True)
            self.assertTrue(all(torch.isfinite(value).all() for value in ck['state_dict'].values()))
            self.assertEqual(ck['fit_people'],['A'])

    def test_primary_fusion_fits_without_changing_experts(self):
        rng=np.random.default_rng(3);n=8
        evidence=dict(ticks=np.arange(n*450).reshape(n,450),imu_logits=np.full((n,450),-.4,np.float32),
            imu_embedding=rng.normal(size=(n,128)).astype(np.float32),imu_support=np.ones((n,450),bool),
            imu_quality=np.ones((n,8),np.float32),imu_role_prob=np.full((n,3),1/3,np.float32),
            ppg_logits=np.full(n,-.2,np.float32),ppg_embedding=rng.normal(size=(n,64)).astype(np.float32),
            ppg_support=np.ones(n,bool),ppg_quality=np.ones(n,np.float32),wave_coverage=np.ones(n,np.float32),
            feature_coverage=np.ones(n,np.float32),target=np.zeros((n,450),np.float32),known=np.ones((n,450),bool))
        evidence['target'][2:5]=1;before=copy.deepcopy(evidence)
        sessions=[dict(primary_evidence=evidence)]
        self.assertEqual(len(FusionEvidenceDataset(sessions,4)),2)
        config=dict(seed=7,epochs=1,batch_size=2,core_frames=4,lr=.001,weight_decay=.001,calibration_penalty=.001,dropout=.1)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fusion.pt';history=train_primary_fusion(sessions,config,path,torch.device('cpu'))
            self.assertEqual(len(history),1);checkpoint=torch.load(path,weights_only=True)
            from eatdet.inference import primary_probabilities
            result=primary_probabilities(evidence,checkpoint)
            self.assertEqual(result['probability'].shape,(n,450));self.assertTrue(np.isfinite(result['probability']).all())
        for key,value in before.items():np.testing.assert_array_equal(value,evidence[key])

    def test_ranker_weights_remain_compatible_with_runtime(self):
        from event_detection.events import generate_candidates, candidate_training_targets
        from event_detection.ranker import predict_ranker
        sessions=[];n=24
        for person in ('A','B'):
            s=dict(ticks=np.arange(n*450).reshape(n,450),imu_logits=np.full((n,450),-4,np.float32),
                imu_support=np.ones((n,450),bool),imu_segment_ids=np.zeros((n,450),np.int64),
                imu_role_prob=np.full((n,2),.5,np.float32),imu_quality=np.ones((n,8),np.float32),
                imu_boundary_logits=np.full((n,2),-3,np.float32),imu_local_logits=np.zeros(n,np.float32),imu_context_logits=np.zeros(n,np.float32),
                ppg_logits=np.full(n,-4,np.float32),ppg_support=np.ones(n,bool),ppg_segment_ids=np.zeros(n,np.int64),
                ppg_start_logits=np.full(n,-3,np.float32),ppg_end_logits=np.full(n,-3,np.float32),
                ppg_quality_raw_max=np.ones(n,np.float32),ppg_quality_expert=np.ones(n,np.float32),
                raw_coverage=np.ones(n,np.float32),spectral_coverage=np.ones(n,np.float32),feature_coverage=np.ones(n,np.float32),
                physical_mask=np.ones((n,450),bool),segment_ids=np.zeros(n,np.int64),known=np.ones((n,450),bool),
                observed=np.ones((n,450),bool),target=np.zeros((n,450),np.float32),
                metadata=dict(person=person,recording_id=person,group='imu_oof_frozen_ppg',imu_oof=True,ppg_oof=False))
            s['imu_logits'][4:12]=3;s['ppg_logits'][18:23]=3;sessions.append(s)
        class Labels:
            def participant_events(self,person):return [dict(participant_key=person,start_epoch_ms=120000,end_epoch_ms=360000)]
        config=dict(architecture='logistic',epochs=2,weight_decay=.01,threshold=.5,hidden=16,dropout=.1,
            lr=.01,batch_size=128,quality_weight=.2,ranking_weight=.1,quality_power=.25,nms_iou=.5,nms_ios=.8,
            candidate_rule=dict(threshold=.4,low_threshold=.15,merge_gap_seconds=60.,min_duration_seconds=90.))
        for session in sessions:
            candidates=generate_candidates(session,config['candidate_rule'])
            target,valid,_,_=candidate_training_targets(session,candidates,Labels().participant_events(session['metadata']['person']))
            self.assertEqual(len(candidates),2)
            self.assertEqual(target[valid].tolist(),[1.,0.])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'ranker.pt';receipt=train_event_ranker(sessions,Labels(),config,path,torch.device('cpu'),17)
            self.assertTrue(receipt['positive'] and receipt['negative']);payload=torch.load(path,weights_only=True)
            candidates=generate_candidates(sessions[0]);scores=predict_ranker(sessions[0],candidates,payload)
            self.assertEqual(scores.shape,(len(candidates),));self.assertTrue(np.isfinite(scores).all())


if __name__=='__main__':unittest.main()
