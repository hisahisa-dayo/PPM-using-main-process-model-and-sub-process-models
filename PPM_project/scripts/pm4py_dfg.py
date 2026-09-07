import numpy as np
import pandas as pd
import pm4py
import datetime  

class ProcessModelGenerator:
    def __init__(self, log, global_act_to_id=None):
        self.log = log
        self.act_to_id = global_act_to_id

    def calculate_transition_matrix(self):
        # 1. logが「リストのリスト」の場合、pm4pyが読み込めるDataFrameに自動変換する
        if isinstance(self.log, list):
            events = []
            base_time = datetime.datetime(2026, 1, 1, 0, 0, 0)
            
            for case_id, trace in enumerate(self.log):
                for i, act in enumerate(trace):
                    events.append({
                        "case:concept:name": str(case_id), 
                        "concept:name": act,
                        "time:timestamp": base_time + datetime.timedelta(seconds=i)
                    })
            df = pd.DataFrame(events)
            
            pm4py_log = pm4py.format_dataframe(
                df, 
                case_id="case:concept:name", 
                activity_key="concept:name",
                timestamp_key="time:timestamp"
            )
        else:
            pm4py_log = self.log

        # 2. 変換したデータを用いてDFGを生成
        dfg, start_activities, end_activities = pm4py.discover_dfg(pm4py_log)

        if self.act_to_id is None:
            activities = set()
            for (from_act, to_act) in dfg.keys():
                activities.add(from_act)
                activities.add(to_act)
            activities.update(start_activities.keys())
            activities.update(end_activities.keys())
            
            self.act_to_id = {act: idx for idx, act in enumerate(sorted(list(activities)))}

        num_acts = len(self.act_to_id)
        matrix = np.zeros((num_acts, num_acts), dtype=float)

        for (from_act, to_act), count in dfg.items():
            if from_act in self.act_to_id and to_act in self.act_to_id:
                matrix[self.act_to_id[from_act], self.act_to_id[to_act]] = float(count)

        for i in range(num_acts):
            row_sum = np.sum(matrix[i])
            if row_sum > 0:
                matrix[i] = matrix[i] / row_sum

        second_order_counts = {}
        if isinstance(self.log, list):
            for trace in self.log:
                for i in range(2, len(trace)):
                    prev_act = trace[i - 2]
                    current_act = trace[i - 1]
                    next_act = trace[i]
                    if (
                        prev_act in self.act_to_id
                        and current_act in self.act_to_id
                        and next_act in self.act_to_id
                    ):
                        key = (self.act_to_id[prev_act], self.act_to_id[current_act])
                        if key not in second_order_counts:
                            second_order_counts[key] = np.zeros(num_acts, dtype=float)
                        second_order_counts[key][self.act_to_id[next_act]] += 1.0

        second_order_model = {}
        for key, counts in second_order_counts.items():
            count_sum = np.sum(counts)
            if count_sum > 0:
                second_order_model[key] = counts / count_sum

        # 💡 修正: 行列と一緒に、pm4pyの可視化に必要なデータも返す
        return matrix, self.act_to_id, dfg, start_activities, end_activities, second_order_model
