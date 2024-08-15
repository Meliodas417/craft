import os
import yaml
import csv
from csv import DictReader
import re

# Util and WidgetUtil are assumed to be in your project, so they are not modified.
from Util import Util
from WidgetUtil import WidgetUtil
from const import LOG_FOLDER


class CompatibleDescriptors:
    src_label = None
    src_id = None
    src_text = None
    target_label = None
    target_id = None
    target_text = None
    src_class = None
    target_class = None


class AbstractEvaluator:
    def __init__(self, technique=None, descriptors_type='default'):
        self.technique = technique
        self.descriptors_type = descriptors_type

    def make_descriptors_compatible(self, row):
        raise NotImplementedError

    def assign_score(self, descriptors):
        raise NotImplementedError

    def get_potential_matches(self, data):
        raise NotImplementedError


class ATMEvaluator(AbstractEvaluator):

    def make_descriptors_compatible(self, row):
        self.add_file_name_to_id(row)
        c_descriptors = CompatibleDescriptors()
        c_descriptors.src_id = row['src_id']
        c_descriptors.target_id = row['target_id']
        c_descriptors.src_text = self.get_text('src', row)
        c_descriptors.target_text = self.get_text('target', row)
        c_descriptors.src_label = self.get_label('src', row)
        c_descriptors.target_label = self.get_label('target', row)
        c_descriptors.src_class = row['src_class']
        c_descriptors.src_class = row['target_class']
        return c_descriptors

    def assign_score(self, descriptors: CompatibleDescriptors):
        if 'EditText' in descriptors.src_class:
            return self.compute_editable(descriptors)
        else:
            return self.compute_non_editable(descriptors)

    def get_potential_matches(self, data):
        threshold_condition = (data[self.technique] >= 0.5)  # Example threshold
        same_type_condition = (data['src_type'] == data['target_type'])
        return data[threshold_condition & same_type_condition]

    def get_text(self, event_side, row):
        if row[event_side + '_text']:
            return row[event_side + '_text']
        if row[event_side + '_content_desc']:
            return row[event_side + '_content_desc']
        if event_side + '_hint' in row:
            return row[event_side + '_hint']
        return ''

    def get_label(self, event_side, row):
        return row[event_side + '_id']

    def compute_editable(self, descriptors: CompatibleDescriptors):
        text_text_score = self.atm_token_sim(descriptors.src_text, descriptors.target_text)
        label_label_score = self.atm_token_sim(descriptors.src_label, descriptors.target_label)
        return max(text_text_score, label_label_score)

    def compute_non_editable(self, descriptors):
        text_text_score = self.atm_token_sim(descriptors.src_text, descriptors.target_text)
        id_id_score = self.atm_token_sim(descriptors.src_id, descriptors.target_id)
        return max(text_text_score, id_id_score * 0.9)

    def atm_token_sim(self, src, target):
        if self.technique.sentence_level:
            return self.technique.calc_sim(src, target)
        else:
            min_length = min(len(src.split()), len(target.split()))
            return self.technique.calc_sim(src, target) * min_length

    def add_file_name_to_id(self, row):
        src_fields, target_fields = ['src_file_name'], ['target_file_name']
        row['src_id'] += ' ' + ' '.join(row[src_fields].to_list())
        row['target_id'] += ' ' + ' '.join(row[target_fields].to_list())


class CraftdroidEvaluator(AbstractEvaluator):

    @staticmethod
    def get_acceptable_targets(x):
        src_class = x['src_class']
        src_type = x['src_type']
        text = str(x['src_text'])
        tgt_classes = [src_class]
        if src_class in ['android.widget.ImageButton', 'android.widget.Button']:
            tgt_classes = ['android.widget.ImageButton', 'android.widget.Button', 'android.widget.TextView']
        elif src_class == 'android.widget.TextView':
            if src_type == 'clickable':
                tgt_classes += ['android.widget.ImageButton', 'android.widget.Button']
                if re.search(r'https://\w+\.\w+', text):
                    tgt_classes.append('android.widget.EditText')
        elif src_class == 'android.widget.EditText':
            tgt_classes.append('android.widget.MultiAutoCompleteTextView')

        elif src_class == 'android.widget.MultiAutoCompleteTextView':
            tgt_classes.append('android.widget.EditText')

        return tgt_classes

    def get_potential_matches(self, data):
        return data[data.apply(lambda x: x['target_class'] in self.get_acceptable_targets(x), axis=1)]

    def assign_score(self, descriptors):
        fields = ['text', 'id', 'content_desc', 'hint', 'parent_text', 'sibling_text']
        w_scores = []
        for attr in fields:
            src_field = 'src_' + attr
            target_field = 'target_' + attr
            if src_field in descriptors and descriptors[src_field] and descriptors[target_field]:
                sim_score = self.technique.calc_sim(descriptors[src_field], descriptors[target_field])
                w_scores.append(sim_score)

        return sum(w_scores) / len(w_scores) if len(w_scores) else 0

    def make_descriptors_compatible(self, row):
        self.add_hint(row)
        return row

    @staticmethod
    def add_hint(row):
        if 'src_hint' in row:
            row['src_text'] = row['src_text'] + ' ' + row['src_hint']
        if 'target_hint' in row:
            row['target_text'] = row['target_text'] + ' ' + row['target_hint']


class Evaluator:
    def __init__(self, sol_file, algorithm='custom'):
        assert os.path.exists(sol_file), "Invalid config file path"
        self.solution = {}
        self.algorithm = algorithm
        with open(sol_file) as f:
            reader = DictReader(f)
            self.solution = [r for r in reader]
        self.res = {'gui': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
                    'oracle': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}}
        self.finished = 0

        # Initialize evaluator based on selected algorithm
        if algorithm == 'atm':
            self.evaluator = ATMEvaluator()
        elif algorithm == 'craftdroid':
            self.evaluator = CraftdroidEvaluator()
        else:
            self.evaluator = AbstractEvaluator()

    def get_all_config_ids(self, sol_file):
        bid = sol_file.split('-')[-1].split('.')[0]
        config_ids = {}
        for row in self.solution:
            aid_from, aid_to = row['aid_from'], row['aid_to']
            if aid_from not in config_ids:
                config_ids[aid_from] = set()
            config_ids[aid_from].add(aid_to)
        res = []
        for k, v_set in config_ids.items():
            for v_ele in v_set:
                res.append('-'.join([k, v_ele, bid]))
        return res

    def evaluate(self, config_id):
        print(f"Evaluating configuration: {config_id}")
        events_from = Util.load_events(config_id, 'base_from')
        events_to = Util.load_events(config_id, 'base_to')
        events_gen = Util.load_events(config_id, 'generated')
        aid_from = config_id.split('-')[0]
        aid_to = config_id.split('-')[1]
        ans = {}
        for row in self.solution:
            if row['aid_from'] == aid_from and row['aid_to'] == aid_to:
                ans[int(row['step_from'])] = int(row['step_to'])
        idx_gen = 0
        events_pred = []
        for idx_from, src_event in enumerate(events_from):
            if idx_gen == len(events_gen):
                break
            while events_gen[idx_gen]['event_type'] == 'stepping':
                events_pred.append(events_gen[idx_gen])
                idx_gen += 1
            events_pred.append(events_gen[idx_gen])
            event_ans = events_to[ans[idx_from]] if ans[idx_from] > -1 \
                else {'class': 'EMPTY_EVENT', 'event_type': src_event['event_type']}
            self.evaluator.judge(events_pred, event_ans, src_event['event_type'])
            events_pred = []
            idx_gen += 1
        if WidgetUtil.is_equal(events_gen[-1], events_to[-1], ignore_activity=True):
            self.finished += 1

        top1, top2, top3, top5 = 0, 0, 0, 0
        mrr = 0.0
        time_taken = 0.0
        zeros_count = 0

        # Save ranking after each evaluation
        self.save_rank(config_id, top1, top2, top3, top5, mrr, time_taken, zeros_count)
        print(f"Rank saved for configuration: {config_id}")

    def output_res(self):
        label = ['tp', 'tn', 'fp', 'fn']
        print(label)
        for k, v in self.res.items():
            print(k)
            res = [v[lbl] for lbl in label]
            print(res)
            print([n / sum(res) for n in res])
            print(f'Precision: {res[0] / (res[0] + res[2])}. '
                  f'Recall: {res[0] / (res[0] + res[3])}')

    def save_rank(self, config_id, top1, top2, top3, top5, mrr, time, zeros):
        rank_file = 'rank.csv'  # 确保这个文件路径是正确的，可以考虑使用绝对路径
        file_exists = os.path.isfile(rank_file)

        with open(rank_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['algorithm', 'descriptors', 'training_set', 'word_embedding',
                                 'top1', 'top2', 'top3', 'top5', 'MRR', 'time', 'zeros'])

            algorithm, descriptors, training_set, word_embedding = config_id.split('-')
            writer.writerow([algorithm, descriptors, training_set, word_embedding,
                             top1, top2, top3, top5, mrr, time, zeros])


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    total = {
        'gui': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0},
        'oracle': {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    }

    solutions = [
        'solution/a1-b11.csv',
        'solution/a1-b12.csv',
        'solution/a2-b21.csv',
        'solution/a2-b22.csv',
        'solution/a3-b31.csv',
        'solution/a3-b32.csv',
        'solution/a4-b41.csv',
        'solution/a4-b42.csv',
        'solution/a5-b51.csv',
        'solution/a5-b52.csv',
    ]

    config = load_config("config.yml")

    for sol in solutions:
        for algo in config['algorithm']:
            evaluator = Evaluator(sol, algorithm=algo)
            for desc in config['descriptors']:
                for train_set in config['train_set']:
                    for embedding in config['active_techniques']:
                        cid = f"{algo}-{desc}-{train_set}-{embedding}"
                        evaluator.evaluate(cid)

        evaluator.output_res()
        print(f'Finished: {evaluator.finished}/{len(solutions)}')
        for event_type in ['gui', 'oracle']:
            for res in ['tp', 'tn', 'fp', 'fn']:
                total[event_type][res] += evaluator.res[event_type][res]

    print('\n*** Total *** ')
    print(total)
    for event_type in ['gui', 'oracle']:
        print(event_type.upper())
        print('Precision:', total[event_type]["tp"] / (total[event_type]["tp"] + total[event_type]["fp"]))
        print('Recall:', total[event_type]["tp"] / (total[event_type]["tp"] + total[event_type]["fn"]))
