















def predicted_events(rows,rule):
    return [e for r in rows for e in decode(r["ticks"],r["probability"],r["supported"],participant_key=r["person"],**rule)]


def evaluation_scope(rows,labels):
    truth=[];coverage=[];background_points=0
    for person in sorted({r["person"] for r in rows}):
        own=[r for r in rows if r["person"]==person]
        ticks=np.concatenate([r["ticks"] for r in own])
        # A native session can start/end inside the same 30-second frame as
        # its neighbour. Count physical ticks once, while decoding separately.
        ticks,ix=np.unique(ticks,return_inverse=True)
        known=np.zeros(len(ticks),bool);observed=known.copy();positive=known.copy()
        np.logical_or.at(known,ix,np.concatenate([r["known"] for r in own]))
        np.logical_or.at(observed,ix,np.concatenate([r["observed"] for r in own]))
        np.logical_or.at(positive,ix,np.concatenate([r["target"]>0 for r in own]))
        background_points+=int((known&~positive).sum())
        for e in labels.participant_events(person):
            inside=(ticks*1000>=e["start_epoch_ms"]*15)&(ticks*1000<e["end_epoch_ms"]*15)
            if not (inside&known).any():continue
            truth.append(Event(person,e["start_epoch_ms"],e["end_epoch_ms"],meal_id=e["event_id"]))
            duration=(e["end_epoch_ms"]-e["start_epoch_ms"])*15/1000
            coverage.append(dict(person=person,event_id=e["event_id"],
                observed_fraction=min(1.,float((inside&observed).sum())/duration),
                known_fraction=min(1.,float((inside&known).sum())/duration)))
    return truth,coverage,background_points


def score(rows,labels,rule,events=None,*,scope=None):
    events=predicted_events(rows,rule) if events is None else events
    predicted=[Event(e["participant_key"],e["start_ms"],e["end_ms"],e["score"]) for e in events]
    truth,coverage,background_points=evaluation_scope(rows,labels) if scope is None else scope
    matched=match_events(predicted,truth,iou_threshold=.25);result=matched["metrics"]
    background_fp=sum(not any(t.subject_id==p.subject_id and t.start_ms<p.end_ms and t.end_ms>p.start_ms for t in truth) for p in predicted)
    hours=background_points/15/3600
    boundary=(result["start_mae_s"]+result["end_mae_s"])/2 if result["TP"] else None
    return dict(result,recall=result["sensitivity"],background_false_positives=background_fp,known_background_hours=hours,
        background_fp_per_hour=background_fp/hours if hours else None,boundary_mae_seconds=boundary,
        detected_meal_ids=sorted(str(t.meal_id) for _,t,_ in matched["matched"]),
        observed_points=sum(int(r["observed"].sum()) for r in rows),
        supported_observed_points=sum(int((r["supported"]&r["observed"]).sum()) for r in rows),
        scope="V1 intersecting: full original meals intersecting known physical observation; strict IoU > 0.25; label-free decoding",
        label_coverage=coverage)


def _rank(report,rule):
    return report["F1"],-report["background_false_positives"],-(report["boundary_mae_seconds"] or 0.),-rule["merge_gap_seconds"]


def select_decoder(rows,labels,config):
    fixed=dict(config["decoder"]);best=fixed;scope=evaluation_scope(rows,labels);report=score(rows,labels,fixed,scope=scope)
    grid=config["decoder_grid"]
    for high,low,gap in product(grid["threshold"],grid["low_threshold"],grid["merge_gap_seconds"]):
        if low>high:continue
        basic=fixed|dict(threshold=high,low_threshold=low,min_duration_seconds=0,merge_gap_seconds=gap)
        candidates=predicted_events(rows,basic)
        for minimum in grid["min_duration_seconds"]:
            rule=basic|dict(min_duration_seconds=minimum)
            def accepted(e):
                short=rule.get("short_min_duration_seconds")
                return (e["_duration_seconds"]>=minimum or (short is not None and e["_duration_seconds"]>=short
                    and e["_high_fraction"]>=rule.get("short_high_fraction",.75)
                    and e["_high_seconds"]>=rule.get("short_high_duration_seconds",60)))
            candidate=score(rows,labels,rule,[e for e in candidates if accepted(e)],scope=scope)
            if _rank(candidate,rule)>_rank(report,best):best,report=rule,candidate
    return best,report



