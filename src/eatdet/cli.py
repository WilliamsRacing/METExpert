"""Portable command-line interface for the submitted model."""
from pathlib import Path
import argparse
import json
import sys

ROOT=Path(__file__).resolve().parents[2]

def _people(value):
    return [p.strip().upper() for p in value.split(',') if p.strip()] if value else None

def main(argv=None):
    parser=argparse.ArgumentParser(description='Multimodal eating-event detection')
    commands=parser.add_subparsers(dest='command',required=True)
    def preparation(p,required=False):
        p.add_argument('--data-root',type=Path,required=required)
        p.add_argument('--work-dir',type=Path,default=ROOT/'.work'/'prepared')
        p.add_argument('--people',help='Comma-separated external IDs; default: all participant folders')
        p.add_argument('--workers',type=int,default=4)
        p.add_argument('--preprocess-workers',type=int,default=1)
        p.add_argument('--device-workers',type=int,default=1)
        p.add_argument('--annotations',type=Path)
        p.add_argument('--split',type=Path)
    p=commands.add_parser('preprocess',help='Prepare competition TXT recordings')
    preparation(p,True)
    p=commands.add_parser('predict',help='Generate the official event CSV using trained weights')
    preparation(p)
    p.add_argument('--prepared',type=Path)
    p.add_argument('--weights',type=Path,default=ROOT/'weights')
    p.add_argument('--output',type=Path,default=ROOT/'results.csv')
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=8)
    p=commands.add_parser('train',help='Train all model components from prepared training data')
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=ROOT/'training_output')
    p.add_argument('--config',type=Path,default=ROOT/'configs'/'training.json')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--workers',type=int)
    p.add_argument('--ppg-checkpoint',type=Path,help='Use a supplied optical checkpoint while fitting downstream components')
    p.add_argument('--smoke',action='store_true',help='Run a short integration training job')
    p=commands.add_parser('check',help='Check all weight hashes and run model forward self-tests')
    p.add_argument('--weights',type=Path,default=ROOT/'weights')
    p=commands.add_parser('annotations',help='Convert official meal tables into training supervision')
    p.add_argument('--meals',type=Path,required=True)
    p.add_argument('--users',type=Path)
    p.add_argument('--secondary',type=Path)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(argv)
    if args.command=='check':
        from .inference import check_models
        result=check_models(args.weights)
    elif args.command=='annotations':
        from .annotations import convert_tables
        tables=[args.meals]+([args.secondary] if args.secondary else [])
        result=convert_tables(tables,args.output_dir,data_root=args.data_root,users_csv=args.users)
    elif args.command in ('preprocess','predict'):
        from .preprocess import prepare, load_prepared
        if args.command=='predict' and args.prepared:
            if args.data_root:parser.error('Choose either --prepared or --data-root')
            prepared=load_prepared(args.prepared)
        else:
            data=args.data_root or ROOT/'ACC_test_data'
            prepared=prepare(data,args.work_dir,annotation_root=args.annotations,split_file=args.split,
                people=_people(args.people),workers=args.workers,device_workers=args.device_workers,
                preprocess_workers=args.preprocess_workers)
        if args.command=='preprocess':result=prepared
        else:
            from .inference import predict
            events=predict(prepared,args.weights,args.output,people=_people(args.people),device=args.device,batch_size=args.batch_size)
            result=dict(completed=True,subjects=len(_people(args.people) or prepared['people']),events=len(events),output=str(args.output.resolve()))
    elif args.command=='train':
        from .training import train_model
        result=train_model(args.prepared,args.output_dir,args.config,device=args.device,workers=args.workers,
            smoke=args.smoke,ppg_checkpoint=args.ppg_checkpoint)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    return 0

if __name__=='__main__':raise SystemExit(main())
