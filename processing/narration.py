import glob, os, json
import numpy as np 
from typing import List, Tuple, Optional, Union, Dict
from tqdm.auto import tqdm 
from processing.text_utils import LibriQuoteUtils, _should_verify_dur, _get_timed_candidates
from functools import partial 
    
class NarrationToQuoteMatcher(LibriQuoteUtils) : 
    """Main class to match quotations with narration segments also present in the same recording.

    Use the `__call__` function to process many books or splits at once.
    Use the `process_one_recording` method to process a single recording.
    Book ids and recording names can be displayed with the following methods, respectively: `list_libriquote_ids` and `list_libriquote_recordings`
    
    arguments:
    `libriquote_path`: str, path to LibriQuote data
    """
    def __init__(
        self,
        libriquote_path:str,
    ):
        super().__init__(libriquote_path)
        # self.default_end_tgt = '<|Q|>'

    def _get_ordered_list(self, align_data) : 
        quotes = [i for i in align_data['quotations']]
        narrs = [i for i in align_data['narrations']]
        full = quotes+narrs
        l = np.asarray([f['start_byte'] for f in full])
        return [full[x] for x in l.argsort()]

    def _verify_duration(self, candidate, min_narr_dur, max_narr_dur) : 
        # Case where both min and max dur are set
        if all([isinstance(max_narr_dur, (int, float)),isinstance(min_narr_dur, (int, float))]) : 
            if all([
                (candidate[1]['cut_end_time'] - candidate[1]['cut_start_time']) <= max_narr_dur,
                (candidate[1]['cut_end_time'] - candidate[1]['cut_start_time']) >= min_narr_dur,
            ]): 
                return True
            else : 
                return False 
        # Case where only max dur set
        elif isinstance(max_narr_dur, (int, float)) : 
            if  (candidate[1]['cut_end_time'] - candidate[1]['cut_start_time']) <= max_narr_dur : 
                return True
            else :
                return False 
        # Case where only min dur set
        elif isinstance(min_narr_dur, (int, float)) : 
            if (candidate[1]['cut_end_time'] - candidate[1]['cut_start_time']) >= min_narr_dur : 
                return True
            else :
                return False 
        # Case where no limit on narration duration
        else : 
            return True
            
    def _get_narration_one_quote(self, curr_quote, candidate_narrs, how='nearest', min_narr_dur=1, max_narr_dur=20) : 
        """Helper function to get the narration for one quote in a book."""        
 
        if how == 'nearest' : 
            if len(candidate_narrs) > 0 : 
                cq_sbyte = curr_quote[1]['start_byte']
                candidate_narrs_bytes = np.asarray([i[1]['end_byte'] for i in candidate_narrs])
                # Sort based on how close the end of the narration is to the start quotation
                sorted_indices = np.argsort(np.abs(np.asarray(candidate_narrs_bytes) - cq_sbyte)) 
                candidate = candidate_narrs[sorted_indices[0]]
            else : 
                return (None,None) 
                
        if how == 'random' :
            if len(candidate_narrs) > 0 : 
                idx = np.random.randint(0, high=len(candidate_narrs))
                candidate = candidate_narrs[idx]
            else : 
                # no valid candidates found
                return (None,None)

        return candidate

    def _process_one_book(self, book_id, how='nearest', min_narr_dur=1, max_narr_dur=20, subset=None):
        """Helper function to match narration segments for all quotes in a single book."""        
        files = [self.recording2qpath[i] for i in self.bid2fs[book_id]]

        all_data = []
        for idx, fl in enumerate(files): 
            with open(fl) as f : 
                l = self._get_ordered_list(json.load(f))
                all_data.extend([(idx, ll) for ll in l])

        quotations = [idx for idx in range(len(all_data)) if all_data[idx][1]['is_quote']]
        out_q = {k:[] for k in files}
        
        # Get candidate narrs per recording files
        candidate_narrs = [[i for i in all_data if all([i[0]==j, not i[1]['is_quote']])] for j in range(len(files))]

        # Discard candidates that does not fullfil the the duration requirements
        is_duration_controlled = _should_verify_dur(min_narr_dur, max_narr_dur)
        if is_duration_controlled:
            candidate_narrs = [_get_timed_candidates(j, min_narr_dur, max_narr_dur) for j in candidate_narrs]

        # Process each quotations
        for qidx in quotations : 
            candidate = self._get_narration_one_quote(all_data[qidx], candidate_narrs[all_data[qidx][0]], how, min_narr_dur, max_narr_dur)
            filename = files[all_data[qidx][0]]
            data = all_data[qidx][1]
            data['narration'] = candidate[1]
            out_q[filename].append(data)
        return book_id, out_q
    
    def process_one_recording(self, recording_name,  how='nearest', min_narr_dur=1, max_narr_dur=20, subset:Dict=None, seed:int=None):
        """Matches narration with all quotes in a single recording. Recording names can be found with the `list_libriquote_recordings` method.
        
        arguments:
        `recording_name`: LibriQuote recording name.
        `how`: str, either one of `nearest` or `random`. If `nearest`, the nearest narration paragraph (in terms of byte distance to the quotation in the original text) that matches the duration requirements will be matched to the quotation. If `random`, a random segment matching the duration requirements will instead be provided.
        `min_narr_dur`: Union[float,int], minimum duration (in seconds) of the matched narration segment.
        `max_narr_dur`: Union[float,int], maximum duration (in seconds) of the matched narration segment.
        `seed`: int, set the random seed for reproducible results when using random matching.
        """
        if seed is not None : 
            assert isinstance(seed, int), "Please provide an integer for the `seed` argument."
        np.random.seed(seed)
        
        
        path = self.recording2qpath[recording_name]
        book_id = path.split('/')[-3]

        with open(path) as f :  
            all_data = self._get_ordered_list(json.load(f))
            all_data = [(0, ll) for ll in all_data]
            
        quotations = [idx for idx in range(len(all_data)) if all_data[idx][1]['is_quote']]
        out_q = {path: []}
        
        # Get candidate narrs per recording files
        candidate_narrs = [[i for i in all_data if all([i[0]==j, not i[1]['is_quote']])] for j in range(1)]

        # Discard candidates that does not fullfil the the duration requirements
        is_duration_controlled = _should_verify_dur(min_narr_dur, max_narr_dur)
        if is_duration_controlled:
            candidate_narrs = [_get_timed_candidates(j, min_narr_dur, max_narr_dur) for j in candidate_narrs]

        if subset is not None :
            quotations = [quotations[k] for k in subset]
        # Process each quotations
        for qidx in quotations : 
            candidate = self._get_narration_one_quote(all_data[qidx], candidate_narrs[all_data[qidx][0]], how, min_narr_dur, max_narr_dur)
            data = all_data[qidx][1]
            data['narration'] = candidate[1]
            out_q[path].append(data)
            
        return book_id, out_q
    
    def __call__(
        self,
        book_ids:Union[str, List[str]]=None,
        split:str=None,
        matching_type:str="nearest",
        min_narr_dur:Union[float,int]=None,
        max_narr_dur:Union[float,int]=None,
        seed:int=None) :
        """Retrieves narration segment for each quote in the provided `book_ids`, or for each book in the provided LibriQuote `split` in parallel. Note that you can find `book_ids` with the `list_libriquote_ids` method.
        The matching process can either be `random` or `nearest` based on their appearance with respect to the quotation. Also, duration restriction on narration segments can be further imposed. Note that if no segment can be found with the requested duration restrictions, this function will return `None` for the specific quotation.
        
        arguments:
        `book_ids`: a string or List of strings matching book identifiers in LibriQuote.
        `split`: a string indicating the LibriQuote split. If `book_ids` is also set, will only process those `book_ids` that comes in the `split`.
        `matching_type`: str, either one of `nearest` or `random`. If `nearest`, the nearest narration paragraph (in terms of byte distance to the quotation in the original text) that matches the duration requirements will be matched to the quotation. If `random`, a random segment matching the duration requirements will instead be provided.
        `min_narr_dur`: Union[float,int], minimum duration (in seconds) of the matched narration segment.
        `max_narr_dur`: Union[float,int], maximum duration (in seconds) of the matched narration segment.
        `seed`: int, set the random seed for reproducible results when using random matching.

        returns:
        `A dictionnary {`book_id`: {`recording_name` : quote_info_with_narration_dict, ...}, ...}
        """
        if seed is not None : 
            assert isinstance(seed, int), "Please provide an integer for the `seed` argument."
        np.random.seed(seed)
        
        if split == 'benchmark' : 
            raise ValueError('Matched narration are already provided in the benchmark data. If you want to modify them please consider do matching with the right quotes in the `test` set.')
        
        if isinstance(book_ids, str) : 
            book_ids = [book_ids]
        
        subset = None
        if split is not None : 
            assert split in ['train', 'train_filt', 'dev'], "`split` argument must be one of LibriQuote splits."
            
            if book_ids is not None : 
                # getting the the right recordings that matches both split and book
                assert isinstance(book_ids, list), "`book_ids` argument must be a list of LibriQuote book identifiers. You can use the `list_libriquote_ids` method to find them."
                split_recordings = set(self.s2fs[split])
                book_recordings =  set(sum([self.bid2fs[i] for i in book_ids],[]))
                processing_ids = split_recordings & book_recordings
                func = self.process_one_recording
                
                if len(processing_ids) == 0 : 
                    raise ValueError(f"Could not find requested book_ids in the following split: {split}")
            else : 
                processing_ids = self.s2fs[split]
                func = self.process_one_recording
                if split == 'train_filt' : 
                    subset = self.train_filtered_ids
        else : 
            assert isinstance(book_ids, list), "`book_ids` argument must be a list of LibriQuote book identifiers. You can use the `list_libriquote_ids` method to find them."
            processing_ids = book_ids
            func = self._process_one_book

        out = {}
        if not subset :
            subset={k : None for k in processing_ids}
            
        for i in tqdm(processing_ids) : 
            res = func(
                i,
                how=matching_type,
                min_narr_dur=min_narr_dur,
                max_narr_dur=max_narr_dur,
                subset=subset[i])

            out[res[0]] = res[1]
        
        return out 