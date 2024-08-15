import time
from copy import deepcopy
import sys
import traceback
import os
from statistics import mean
import pickle
from collections import defaultdict
import math
from datetime import datetime
import csv
import logging
import yaml


# Local imports
from Util import Util
from StrUtil import StrUtil
from Configuration import Configuration
from Runner import Runner
from WidgetUtil import WidgetUtil
from CallGraphParser import CallGraphParser
from ResourceParser import ResourceParser
from const import SA_INFO_FOLDER, SNAPSHOT_FOLDER

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class Explorer:
    def __init__(self, config_id, appium_port='4723', udid=None):
        self.config = Configuration(config_id)
        self.runner = Runner(self.config.pkg_to, self.config.act_to, self.config.no_reset, appium_port, udid)
        self.src_events = Util.load_events(self.config.id, 'base_from')
        self.tid = self.config.id
        self.current_src_index = 0
        self.tgt_events = []
        self.f_target = 0
        self.prev_tgt_events = []
        self.f_prev_target = -1
        self.rp = ResourceParser(os.path.join(SA_INFO_FOLDER, self.config.id.split('-')[1]))
        self.widget_db = self.generate_widget_db()
        self.cgp = CallGraphParser(os.path.join(SA_INFO_FOLDER, self.config.id.split('-')[1]))
        self.invalid_events = defaultdict(list)
        self.nearest_button_to_text = None
        self.idx_src_to_tgt = {}
        self.skipped_match = defaultdict(list)
        self.consider_naf_only_widget = False

        logging.basicConfig(filename='explorer.log', level=logging.INFO)

    def generate_widget_db(self):
        db = {}
        for w in self.rp.get_widgets():
            if w['activity']:
                w_signature = WidgetUtil.get_widget_signature(w)
                db[w_signature] = w
        return db

    def mutate_src_action(self, mutant):
        for e in self.src_events:
            if e['action'][0] in mutant:
                e['action'][0] = mutant[e['action'][0]]

    def run(self):
        config = load_config("config.yml")
        combinations = [
            (algo, desc, train_set, embedding)
            for algo in config['algorithm']
            for desc in config['descriptor']
            for train_set in config['training_dataset']
            for embedding in config['word_embedding']
        ]

        for algo, desc, train_set, embedding in combinations:
            # Set up the configuration ID based on the combination
            full_config_id = f"{algo}-{desc}-{train_set}-{embedding}-{self.config.id}"

            logging.info(f"Running with configuration: {full_config_id}")

            self.f_prev_target = 0
            self.f_target = 0
            self.tgt_events = []
            self.prev_tgt_events = []
            self.current_src_index = 0
            self.invalid_events = defaultdict(list)
            self.skipped_match = defaultdict(list)
            self.idx_src_to_tgt = {}
            is_explored = False

            # Main exploration loop
            while self.f_target - self.f_prev_target > 0.001:
                logging.info('--\nStart a new round to find a better tgt event sequence')
                logging.info(f'Timestamp: {datetime.now()}')
                self.f_prev_target = self.f_target
                self.prev_tgt_events = self.tgt_events
                self.tgt_events = []
                self.current_src_index = 0

                while self.current_src_index < len(self.src_events):
                    src_event = self.src_events[self.current_src_index]
                    logging.info(f'Source Event:\n{src_event}')

                    if self.current_src_index == len(self.src_events) - 1 and src_event['event_type'] == "oracle":
                        self.consider_naf_only_widget = True
                    else:
                        self.consider_naf_only_widget = False

                    tgt_event = None
                    if self.current_src_index > 0:
                        prev_src_event = self.src_events[self.current_src_index - 1]
                        if prev_src_event['event_type'] == 'oracle' and src_event['event_type'] == 'gui' \
                                and WidgetUtil.is_equal(prev_src_event, src_event) \
                                and self.tgt_events[-1]['class'] != 'EMPTY_EVENT':
                            tgt_event = deepcopy(self.tgt_events[-1])
                            if 'stepping_events' in tgt_event:
                                tgt_event['stepping_events'] = []
                            tgt_event['event_type'] = 'gui'
                            tgt_event['action'] = deepcopy(src_event['action'])
                            if self.check_skipped(tgt_event):
                                tgt_event = None

                    if src_event['event_type'] == 'SYS_EVENT':
                        tgt_event = deepcopy(src_event)

                    backtrack = False
                    if not tgt_event:
                        try:
                            dom, pkg, act = self.execute_target_events([])
                        except:
                            logging.error(f'Backtrack to the previous step due to an exception in execution.')
                            invalid_event = self.tgt_events[-1]
                            self.current_src_index -= 1
                            if self.current_src_index == 0:
                                self.tgt_events = []
                            else:
                                self.tgt_events = self.tgt_events[:self.idx_src_to_tgt[self.current_src_index - 1] + 1]
                            self.invalid_events[self.current_src_index].append(deepcopy(invalid_event))
                            continue

                        self.cache_seen_widgets(dom, pkg, act)

                        w_candidates = []
                        num_to_check = 10
                        if src_event['action'][0] == 'wait_until_text_invisible':
                            if not self.nearest_button_to_text:
                                tgt_event = Explorer.generate_empty_event(src_event['event_type'])
                            else:
                                w_candidates = WidgetUtil.most_similar(self.nearest_button_to_text, self.widget_db.values(),
                                                                       self.config.use_stopwords,
                                                                       self.config.expand_btn_to_text,
                                                                       self.config.cross_check)
                                num_to_check = 1
                        else:
                            w_candidates = WidgetUtil.most_similar(src_event, self.widget_db.values(),
                                                                   self.config.use_stopwords,
                                                                   self.config.expand_btn_to_text,
                                                                   self.config.cross_check)

                        for i, (w, _) in enumerate(w_candidates[:num_to_check]):
                            logging.info(f'({i + 1}/{num_to_check}) Validating Similar w: {w}'.encode("utf-8").decode("utf-8"))
                            if any([WidgetUtil.is_equal(w, e) for e in self.invalid_events.get(self.current_src_index, [])]):
                                logging.info('Skip a known broken event:', w)
                                continue
                            if src_event['action'][0] == 'wait_until_element_presence':
                                is_empty_atc = False
                                attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'clickable', 'password', 'naf'})
                                for atc in attrs_to_check:
                                    if not w[atc]:
                                        atc_in_oracle = 'id' if atc == 'resource-id' else atc
                                        if src_event['action'][2] == atc_in_oracle:
                                            is_empty_atc = True
                                            break
                                        elif src_event['action'][2] == 'xpath' and '@' + atc in src_event['action'][3]:
                                            is_empty_atc = True
                                            break
                                if is_empty_atc:
                                    logging.info('Skip the widget without the attribute that the action is waiting for')
                                    continue
                            try:
                                match = self.check_reachability(w, pkg, act)
                            except Exception as excep:
                                logging.error(excep)
                                traceback.print_exc()
                                return False, self.current_src_index
                            if match:
                                if match['class'] == 'android.widget.EditText' and 'send_keys' in src_event['action'][0]:
                                    if self.check_skipped(match):
                                        logging.info(f'Duplicated match (later): {match}\n. Skipped.')
                                        continue
                                    is_mapped, tgt_idx, src_idx = self.check_mapped(match)
                                    is_idential_src_widgets = self.check_identical_src_widgets(src_idx, self.current_src_index)
                                    if is_mapped and not is_idential_src_widgets:
                                        if match['score'] <= self.tgt_events[tgt_idx]['score']:
                                            logging.info(f'Duplicated match (previous): {match}\n. Skipped.')
                                            continue
                                        else:
                                            logging.info(f'Duplicated match. Backtrack to src_idx: {src_idx} to find another match')
                                            backtrack = True
                                            self.current_src_index = src_idx
                                            self.skipped_match[src_idx].append(deepcopy(self.tgt_events[tgt_idx]))
                                            if src_idx == 0:
                                                self.tgt_events = []
                                            else:
                                                self.tgt_events = self.tgt_events[:self.idx_src_to_tgt[src_idx - 1] + 1]
                                            break
                                if 'clickable' not in w:
                                    self.widget_db.pop(WidgetUtil.get_widget_signature(w), None)
                                if src_event['action'][0] == 'wait_until_text_invisible':
                                    if self.runner.check_text_invisible(src_event):
                                        tgt_event = self.generate_event(match, deepcopy(src_event['action']))
                                    else:
                                        tgt_event = Explorer.generate_empty_event(src_event['event_type'])
                                else:
                                    tgt_event = self.generate_event(match, deepcopy(src_event['action']))
                                break
                    if backtrack:
                        continue

                    if not tgt_event:
                        tgt_event = Explorer.generate_empty_event(src_event['event_type'])

                    if tgt_event['class'] == 'EMPTY_EVENT' and tgt_event['event_type'] == 'oracle' and not is_explored:
                        logging.info('Empty event for an oracle. Try to explore the app')
                        self.reset_and_explore(self.tgt_events)
                        is_explored = True
                        continue
                    else:
                        is_explored = False

                    logging.info('** Learned for this step:')
                    if 'stepping_events' in tgt_event and tgt_event['stepping_events']:
                        self.tgt_events += tgt_event['stepping_events']
                        for t in tgt_event['stepping_events']:
                            logging.info(t)
                    logging.info(tgt_event)
                    logging.info('--')
                    self.tgt_events.append(tgt_event)
                    self.idx_src_to_tgt[self.current_src_index] = len(self.tgt_events) - 1
                    self.current_src_index += 1

                self.f_target = self.fitness(self.tgt_events)
                logging.info(f'Current target events with fitness {self.f_target}:')
                for t in self.tgt_events:
                    logging.info(t)
                self.snapshot()

                if self.f_target == self.f_prev_target == 0:
                    logging.info('All Empty Events. Explore the app and start over.')
                    self.reset_and_explore()
                    self.tgt_events = []
                    self.prev_tgt_events = []
                    self.f_prev_target = -1

            # Save the results to rank.csv
            self.save_rank(full_config_id, mrr=0, time=time.time() - t_start, zeros=0)

        return True, 0

    def reset_and_explore(self, tgt_events=[]):
        self.runner.perform_actions(tgt_events, reset=True)
        all_widgets = WidgetUtil.find_all_widgets(self.runner.get_page_source(),
                                                  self.runner.get_current_package(),
                                                  self.runner.get_current_activity(),
                                                  self.config.pkg_to)
        btn_widgets = []
        for w in all_widgets:
            if w['class'] in ['android.widget.Button', 'android.widget.ImageButton', 'android.widget.TextView']:
                attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'class', 'clickable', 'password'})
                attr_check = [attr in w and w[attr] for attr in attrs_to_check]
                if w['clickable'] == 'true' and any(attr_check):
                    btn_widgets.append(w)
        for btn_w in btn_widgets:
            self.runner.perform_actions(tgt_events, reset=True)
            btn_w['action'] = ['click']
            self.runner.perform_actions([btn_w], reset=False, cgp=self.cgp)
            self.cache_seen_widgets(self.runner.get_page_source(),
                                    self.runner.get_current_package(),
                                    self.runner.get_current_activity())

    def cache_seen_widgets(self, dom, pkg, act):
        current_widgets = WidgetUtil.find_all_widgets(dom, pkg, act, self.config.pkg_to)
        for w in current_widgets:
            w_signature = WidgetUtil.get_widget_signature(w)
            w_sa = {k: v for k, v in w.items() if k not in ['clickable', 'password']}
            w_sa_signature = WidgetUtil.get_widget_signature(w_sa)
            popped = self.widget_db.pop(w_sa_signature, None)
            if popped:
                logging.info('** wDB (SA) popped:', popped)
            tmp_email = self.runner.databank.get_temp_email(renew=False)
            if tmp_email in w_signature:
                pre = w_signature.split(tmp_email)[0]
                if not pre.endswith('!'):
                    pre = pre.replace(pre.split('!')[-1], '', 1)
                post = w_signature.split(tmp_email)[-1]
                if not post.startswith('!'):
                    post = post.replace(post.split('!')[0], '', 1)
                discarded_keys = []
                for k in self.widget_db.keys():
                    if k.startswith(pre) and k.endswith(post) and k != pre + post:
                        if StrUtil.is_contain_email(self.widget_db[k]['text']):
                            discarded_keys.append(k)
                for k in discarded_keys:
                    popped = self.widget_db.pop(k, None)
                    if popped:
                        logging.info('** wDB (obsolete Email) popped:', popped)
            self.widget_db[w_signature] = w

    def execute_target_events(self, stepping_events):
        src_event = self.src_events[self.current_src_index]
        require_wait = src_event['action'][0].startswith('wait_until')
        self.runner.perform_actions(self.tgt_events, require_wait, reset=True, cgp=self.cgp)
        self.runner.perform_actions(stepping_events, require_wait, reset=False, cgp=self.cgp)
        return self.runner.get_page_source(), self.runner.get_current_package(), self.runner.get_current_activity()

    @staticmethod
    def generate_event(w, actions=None):
        if actions[0] == 'wait_until_element_presence':
            if actions[2] == 'xpath' and '@content-desc=' in actions[3]:
                pre, post = actions[3].split('@content-desc=')
                post = f'@content-desc="{w["content-desc"]}"' + ''.join(post.split('"')[2:])
                actions[3] = pre + post
            elif actions[2] == 'xpath' and '@text=' in actions[3]:
                pre, post = actions[3].split('@text=')
                post = f'@text="{w["text"]}"' + ''.join(post.split('"')[2:])
                actions[3] = pre + post
            elif actions[2] == 'xpath' and 'contains(@text,' in actions[3]:
                pre, post = actions[3].split('contains(@text,')
                post = f'contains(@text, "{w["text"]}"' + ''.join(post.split('"')[2:])
                actions[3] = pre + post
            elif actions[2] == 'id':
                actions[3] = w['resource-id']
        w['action'] = actions
        return w

    @staticmethod
    def generate_empty_event(event_type):
        return {"class": "EMPTY_EVENT", 'score': 0, 'event_type': event_type}

    def check_reachability(self, w, current_pkg, current_act):
        act_from = current_pkg + current_act
        act_to = w['package'] + w['activity']
        potential_paths = self.cgp.get_paths_between_activities(act_from, act_to, self.consider_naf_only_widget)
        if w['activity'] == current_act:
            potential_paths.insert(0, [])
        logging.info(f'Activity transition: {act_from} -> {act_to}. {len(potential_paths)} paths to validate.')
        invalid_paths = []
        for ppath in potential_paths:
            match = self.validate_path(ppath, w, invalid_paths)
            if match:
                return match
        return None

    def validate_path(self, ppath, w_target, invalid_paths):
        path_show = []
        for hop in ppath:
            if '(' in hop:
                if hop.startswith('D@'):
                    gui = ' '.join(hop.split()[:-1])
                else:
                    gui = self.rp.get_wName_from_oId(hop.split()[0])
                path_show.append(gui)
            else:
                path_show.append(StrUtil.get_activity((hop)))
        logging.info(f'Validating path: ', path_show)
        for ip in invalid_paths:
            if ip == ppath[:len(ip)]:
                logging.info('Known invalid path prefix:', ppath[:len(ip)])
                return None
        _, __, ___ = self.execute_target_events([])
        stepping = []
        for i, hop in enumerate(ppath):
            if '(' in hop:
                w_id = ' '.join(hop.split()[:-1])
                action = hop.split('(')[1][:-1]
                action = 'long_press' if action in ['onItemLongClick', 'onLongClick'] else 'click'
                if w_id.startswith('D@'):
                    kv_pairs = w_id[2:].split('&')
                    kv = [kvp.split('=') for kvp in kv_pairs]
                    criteria = {k: v for k, v in kv}
                    logging.info('D@criteria:', criteria)
                    w_stepping = WidgetUtil.locate_widget(self.runner.get_page_source(), criteria)
                else:
                    w_name = self.rp.get_wName_from_oId(w_id)
                    w_stepping = WidgetUtil.locate_widget(self.runner.get_page_source(), {'resource-id': w_name})
                if not w_stepping:
                    is_existed = False
                    for ip in invalid_paths:
                        if ip == ppath[:i + 1]:
                            is_existed = True
                    if not is_existed:
                        invalid_paths.append([h for h in ppath[:i + 1]])
                    return None
                w_stepping['action'] = [action]
                w_stepping['activity'] = self.runner.get_current_activity()
                w_stepping['package'] = self.runner.get_current_package()
                w_stepping['event_type'] = 'stepping'
                stepping.append(w_stepping)
                act_from = self.runner.get_current_package() + self.runner.get_current_activity()
                self.runner.perform_actions([stepping[-1]], require_wait=False, reset=False, cgp=self.cgp)
                self.cache_seen_widgets(self.runner.get_page_source(),
                                        self.runner.get_current_package(),
                                        self.runner.get_current_activity())
                act_to = self.runner.get_current_package() + self.runner.get_current_activity()
                self.cgp.add_edge(act_from, act_to, w_stepping)
        attrs_to_check = set(WidgetUtil.FEATURE_KEYS).difference({'clickable', 'password'})
        criteria = {k: w_target[k] for k in attrs_to_check if k in w_target}
        if self.src_events[self.current_src_index]['action'][0] == 'wait_until_text_presence':
            criteria['text'] = self.src_events[self.current_src_index]['action'][3]
        if self.current_src_index > 0 and self.is_for_email_or_pwd(self.src_events[self.current_src_index - 1],
                                                                   self.src_events[self.current_src_index]):
            if StrUtil.is_contain_email(self.src_events[self.current_src_index]['action'][1]):
                criteria['text'] = self.runner.databank.get_temp_email(renew=False)
        w_tgt = WidgetUtil.locate_widget(self.runner.get_page_source(), criteria)
        if not w_tgt:
            return None
        else:
            src_event = self.src_events[self.current_src_index]
            w_tgt['stepping_events'] = stepping
            w_tgt['package'] = self.runner.get_current_package()
            w_tgt['activity'] = self.runner.get_current_activity()
            w_tgt['event_type'] = src_event['event_type']
            w_tgt['score'] = WidgetUtil.weighted_sim(w_tgt, src_event)
            if src_event['action'][0] == 'wait_until_text_invisible':
                for k in w_tgt.keys():
                    if k not in ['stepping_events', 'package', 'activity', 'event_type', 'score']:
                        w_tgt[k] = ''
            if src_event['action'][0] == 'wait_until_text_presence':
                self.nearest_button_to_text = WidgetUtil.get_nearest_button(self.runner.get_page_source(), w_tgt)
                self.nearest_button_to_text['activity'] = w_tgt['package']
                self.nearest_button_to_text['package'] = w_tgt['activity']
            return w_tgt

    @staticmethod
    def fitness(events):
        gui_scores = [float(e['score']) for e in events if e['event_type'] == 'gui']
        oracle_scores = [float(e['score']) for e in events if e['event_type'] == 'oracle' and e['score'] is not None]
        gui = mean(gui_scores) if gui_scores else 0
        oracle = mean(oracle_scores) if oracle_scores else 0
        return 0.5 * gui + 0.5 * oracle

    def save_rank(self, config_id, mrr, time, zeros):
        rank_file = 'rank.csv'
        file_exists = os.path.isfile(rank_file)

        with open(rank_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(
                    ['algorithm', 'descriptors', 'app_id', 'MRR', 'time' ])

            algorithm, descriptors, training_set, word_embedding, *app_id_parts = config_id.split('-')
            app_id = '-'.join(app_id_parts)
            writer.writerow([algorithm, descriptors, training_set, word_embedding, app_id, mrr, time, zeros])

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['runner']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.runner = None

    def snapshot(self):
        with open(os.path.join(SNAPSHOT_FOLDER, self.config.id + '.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def check_mapped(self, match):
        tgt_idx = -1
        for i, e in enumerate(self.tgt_events):
            if e['class'] != 'android.widget.EditText' or 'send_keys' not in e['action'][0]:
                continue
            e_tgt_new_text = deepcopy(e)
            e_tgt_new_text['text'] = e_tgt_new_text['action'][1]
            if WidgetUtil.is_equal(match, e) or WidgetUtil.is_equal(match, e_tgt_new_text):
                tgt_idx = i
                break
        if tgt_idx == -1:
            return False, -1, -1
        else:
            src_idx = -1
            for i_src, i_tgt in self.idx_src_to_tgt.items():
                if i_tgt == tgt_idx:
                    src_idx = i_src
                    break
            assert src_idx != -1
            return True, tgt_idx, src_idx

    def check_skipped(self, match):
        for skipped in self.skipped_match[self.current_src_index]:
            skipped_new_text = deepcopy(skipped)
            skipped_new_text['text'] = skipped_new_text['action'][1]
            if WidgetUtil.is_equal(match, skipped) or WidgetUtil.is_equal(match, skipped_new_text):
                return True
        return False

    def check_identical_src_widgets(self, src_idx1, src_idx2):
        if src_idx1 == -1 or src_idx2 == -1:
            return True
        src_e1 = self.src_events[src_idx1]
        src_e2 = self.src_events[src_idx2]
        src_classes_to_check = ['android.widget.EditText', 'android.widget.MultiAutoCompleteTextView']
        if src_e1['class'] in src_classes_to_check and src_e2['class'] in src_classes_to_check:
            if self.is_for_email_or_pwd(src_e1, src_e2):
                return True
            else:
                w1 = deepcopy(src_e1)
                w1['text'] = ''
                w2 = deepcopy(src_e2)
                w2['text'] = ''
                return WidgetUtil.is_equal(w1, w2)
        else:
            return True

    def is_for_email_or_pwd(self, src_e1, src_e2):
        if 'send_keys' in src_e1['action'][0] and 'send_keys' in src_e2['action'][0]:
            if src_e1['action'][1] == src_e2['action'][1]:
                if StrUtil.is_contain_email(src_e1['action'][1]) or \
                        src_e1['action'][1] == self.runner.databank.get_password():
                    return True
        return False

    def decay_by_distance(self, w_candidates, current_pkg, current_act):
        new_candidates = []
        for w, score in w_candidates:
            act_from = current_pkg + current_act
            act_to = w['package'] + w['activity']
            if act_from == act_to:
                d = 1
            else:
                potential_paths = self.cgp.get_paths_between_activities(act_from, act_to, self.consider_naf_only_widget)
                if not potential_paths:
                    d = 2
                else:
                    shortest_path, shortest_d = potential_paths[0], len(potential_paths[0])
                    for ppath in potential_paths[1:]:
                        if len(ppath) < shortest_d:
                            shortest_path, shortest_d = ppath, len(ppath)
                    d = len([hop for hop in shortest_path if '(' in hop or 'D@' in hop])
                    assert d >= 1
            new_score = score / (1 + math.log(d, 2))
            new_candidates.append((w, new_score))
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        if [s for w, s in w_candidates[:10]] != [s for w, s in new_candidates[:10]]:
            logging.info('** Similarity rank changed after considering distance')
        return new_candidates


if __name__ == '__main__':
    if len(sys.argv) > 1:
        app_id = sys.argv[1]  # This should be in the format "a21-a22-b21"
        appium_port = sys.argv[2]
        udid = sys.argv[3]
    else:
        app_id = 'a33-a35-b31'
        appium_port = '5723'
        udid = 'emulator-5556'

    explorer = Explorer(app_id, appium_port, udid)

    t_start = time.time()
    is_done, failed_step = explorer.run()
    if is_done:
        logging.info('Finished. Learned actions')
        if explorer.f_prev_target > explorer.f_target:
            results = explorer.prev_tgt_events
        else:
            results = explorer.tgt_events
        for a in results:
            logging.info(a)
        logging.info(f'Transfer time in sec: {time.time() - t_start}')
        logging.info('Start testing learned actions')
        t_start = time.time()
        try:
            explorer.runner.perform_actions(results)
            logging.info(f'Testing time in sec: {time.time() - t_start}')
        except Exception as excep:
            logging.error(f'Error when validating learned actions\n{excep}')
    else:
        logging.info(f'Failed transfer at source index {failed_step}')
        logging.info(f'Transfer time in sec: {time.time() - t_start}')
        results = explorer.tgt_events
    Util.save_events(results, app_id)
