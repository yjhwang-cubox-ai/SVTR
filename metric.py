from rapidfuzz.distance import Levenshtein
from difflib import SequenceMatcher

class OneMinusNEDMetric:
    def __init__(self):
        self.results = []

    def process(self, infer_results):
        for result in infer_results:
            gt_text = result['gt_text']
            pred_text = result['pred_text']
            
            gt_text_lower = gt_text.lower()
            pred_text_lower = pred_text.lower()
            #normalized_distace 에 소문자 변환 적용X
            norm_ed = Levenshtein.normalized_distance(pred_text,
                                                      gt_text)
            result = dict(img=result['img'], norm_ed=norm_ed)
            self.results.append(result)

    def compute_metrics(self, infer_results):
        self.process(infer_results)
                
        gt_word_num = len(self.results)
        norm_ed = [result['norm_ed'] for result in self.results]
        norm_ed_sum = sum(norm_ed)
        normalized_edit_distance = norm_ed_sum / max(1, gt_word_num)
        eval_res = {}
        eval_res['1-N.E.D'] = 1.0 - normalized_edit_distance
        for key, value in eval_res.items():
            eval_res[key] = float(f'{value:.4f}')
        return eval_res

class WordMetric:
    def __init__(self):
        self.results = []
        
    def process(self, infer_results):
        for result in infer_results:
            match_num = 0
            match_ignore_case_num = 0
            match_ignore_case_symbol_num = 0
            pred_text = result['pred_text']
            gt_text = result['gt_text']
            
            pred_text_lower = pred_text.lower()
            gt_text_lower = gt_text.lower()
            
            match_ignore_case_num = pred_text_lower == gt_text_lower
            match_num = pred_text == gt_text
            
            #ToDo(05.14): match_ignore_case_symbol_num 구현
            
            result = dict(
                match_num=match_num,
                match_ignore_case_num=match_ignore_case_num
            )
            
            self.results.append(result)

    def compute_metrics(self, infer_results):
        self.process(infer_results)

        eps = 1e-8
        eval_res = {}
        gt_word_num = len(self.results)
        
        match_nums = [result['match_num'] for result in self.results]
        match_nums = sum(match_nums)
        eval_res['word_acc'] = 1.0 * match_nums / (eps + gt_word_num)        
        
        match_ignore_case_num = [
            result['match_ignore_case_num'] for result in self.results
        ]
        match_ignore_case_num = sum(match_ignore_case_num)
        eval_res['word_acc_ignore_case'] = 1.0 *\
            match_ignore_case_num / (eps + gt_word_num)

        for key, value in eval_res.items():
            eval_res[key] = float(f'{value:.4f}')
        return eval_res

class CharMetric:
    def __init__(self):
        self.results = []

    def process(self, infer_results):
        for result in infer_results:
            pred_text = result['pred_text']
            gt_text = result['gt_text']
            gt_text_lower = gt_text.lower()
            pred_text_lower = pred_text.lower()
            
            # gt_text_lower_ignore = self.valid_symbol.sub('', gt_text_lower)
            # pred_text_lower_ignore = self.valid_symbol.sub('', pred_text_lower)
            # number to calculate char level recall & precision
            
            # ToDo(05.14): 원본 평가 metric 에는 gt_text_lower_ignore 와 pred_text_lower_ignore 를 사용했었음
            # 추후 수정해야함
            
            result = dict(
                gt_char_num=len(gt_text_lower),
                pred_char_num=len(pred_text_lower),
                true_positive_char_num=self._cal_true_positive_char(
                    pred_text_lower, gt_text_lower))
            self.results.append(result)

    def compute_metrics(self, infer_results):
        self.process(infer_results)
        gt_char_num = [result['gt_char_num'] for result in self.results]
        pred_char_num = [result['pred_char_num'] for result in self.results]
        true_positive_char_num = [
            result['true_positive_char_num'] for result in self.results
        ]
        gt_char_num = sum(gt_char_num)
        pred_char_num = sum(pred_char_num)
        true_positive_char_num = sum(true_positive_char_num)

        eps = 1e-8
        char_recall = 1.0 * true_positive_char_num / (eps + gt_char_num)
        char_precision = 1.0 * true_positive_char_num / (eps + pred_char_num)
        eval_res = {}
        eval_res['char_recall'] = char_recall
        eval_res['char_precision'] = char_precision

        for key, value in eval_res.items():
            eval_res[key] = float(f'{value:.4f}')
        return eval_res

    def _cal_true_positive_char(self, pred: str, gt: str) -> int:
        
        """Calculate correct character number in prediction.

        Args:
            pred (str): Prediction text.
            gt (str): Ground truth text.

        Returns:
            true_positive_char_num (int): The true positive number.
        """

        all_opt = SequenceMatcher(None, pred, gt)
        true_positive_char_num = 0
        for opt, _, _, s2, e2 in all_opt.get_opcodes():
            if opt == 'equal':
                true_positive_char_num += (e2 - s2)
            else:
                pass
        return true_positive_char_num