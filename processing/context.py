import glob, os, json
import numpy as np 
from typing import List, Tuple, Optional, Union
from multiprocessing import Pool
from tqdm.auto import tqdm 
import spacy
from processing.text_utils import LibriQuoteUtils, _segment_into_paragraph, is_quote_punct
from functools import partial 

class ContextRetriever(LibriQuoteUtils) : 
    """Main class to retrieve contextual information for quotes in books. By default, will match the quote with its position in the original book text, and will assign contextual information as a number of paragraphs before and after the quotation.

    arguments:
    `libriquote_path`: str, path to LibriQuote data
    `max_lookup_paragraphs`: int, maximum number of paragraphs used to lookup the potential contextual paragraphs. Set this higher is you inted to derive very large context windows.
    `margin`: float, a margin (in percentage) to allow the inclusion of a new paragraph it does not excede the maximum number of tokens allowed times the margin.
    """
    def __init__(
        self,
        libriquote_path:str,
        max_lookup_paragraphs:int = 10,
        margin:float=0.1
    ):
        super().__init__(libriquote_path)
        self.nlp = spacy.load('en_core_web_sm',disable=["tok2vec", "ner", "tagger", "attribute_ruler", "lemmatizer", "parser"])
        self.max_lookup_paragraphs = max_lookup_paragraphs
        self.margin = 1 + margin
        self.default_tgt_marker = '<|Q|>'
        # self.default_end_tgt = '<|Q|>'

    def _get_ordered_list(self, align_data) : 
        quotes = [i for i in align_data['quotations']]
        narrs = [i for i in align_data['narrations']]
        full = quotes+narrs
        l = np.asarray([f['start_byte'] for f in full])
        return [full[x] for x in l.argsort()]
       
    def _get_context_one_quote(self, index, all_data, orig_text, num_tokens, insert_tgt_markers) : 
        """Helper function to get the context of one quote in a book."""        
        curr_quote = all_data[index]
        st_tgt_byte, e_tgt_byte = curr_quote[-1]['start_byte'], curr_quote[-1]['end_byte'] +1 
        try : 
            if not is_quote_punct(orig_text[st_tgt_byte]): 
                st_tgt_byte -= 1
            if not is_quote_punct(orig_text[e_tgt_byte]): 
                if orig_text[e_tgt_byte-1] in ['\n', ' '] :
                    e_tgt_byte -=1 
        except Exception as e : 
            print(f'Could not find positions ({st_tgt_byte}, {e_tgt_byte}) in the text (length={len(orig_text)})')
            return index, None
            
        # start paragraph
        spidx = max(0, index-self.max_lookup_paragraphs)
        # insuring same recording
        right_pars = [i for i in all_data[spidx:index] if i[0] == curr_quote[0]]
        if len(right_pars) == 0 :
            # case where no right context
            context_start_byte = st_tgt_byte
        else : 
            offset = right_pars[0][1]['start_byte']
            # get associated text and slit into paragraphs using \n\n
            right_pars, indices = _segment_into_paragraph(orig_text[right_pars[0][1]['start_byte']: right_pars[-1][1]['end_byte']])
            # perform tokenization to get number of tokens per paragraph
            tokens = [len(self.nlp.tokenizer(j)) for j in right_pars]
            # Iterate to get the right amount of paragraphs with around max `num_tokens` tokens
            cnt = 0 
            for idx in reversed(range(len(tokens))) : 
                
                length = tokens[idx]
                cnt += length
                if all([(
                    idx != len(right_pars) - 1), # we take first paragraph anyway
                    (cnt > (num_tokens * self.margin)) # we allow a some margin for number of tokens
                       ]):
                    break
            context_start_byte = offset + indices[min(len(tokens)-1, idx+1)]['st']

        # same for end paragraph
        epidx = min(len(all_data), index+self.max_lookup_paragraphs+1)
        # insuring same recording
        end_pars = [i for i in all_data[index+1:epidx] if i[0] == curr_quote[0]]
        if len(end_pars) == 0 :
            # case where no left context
            context_end_byte = e_tgt_byte
        else : 
            offset = end_pars[0][1]['start_byte']
            # get associated text and slit into paragraphs using \n\n
            end_pars, indices = _segment_into_paragraph(orig_text[end_pars[0][1]['start_byte']: end_pars[-1][1]['end_byte']])
            # perform tokenization to get number of tokens per paragraph
            tokens = [len(self.nlp.tokenizer(j)) for j in end_pars]
            # Iterate to get the right amount of paragraphs with around max `num_tokens` tokens
            # final_eps = []
            cnt = 0 
            for idx in range(len(tokens)) : 
                length = tokens[idx]
                cnt += length
                if all([(
                    idx != 0), # we take first paragraph anyway
                    (cnt > (num_tokens * self.margin)) # we allow a some margin for number of tokens
                       ]):
                    break
                    
            context_end_byte = offset + indices[max(0,idx-1)]['end']

        # If asked, we insert start and end markers to the target quotation for easier identification 
        if isinstance(insert_tgt_markers,bool) :
            if insert_tgt_markers : 
                st_marker = end_marker = self.default_tgt_marker
            else : 
                return orig_text[context_start_byte:context_end_byte]
                
        elif isinstance(insert_tgt_markers, str) : 
            st_marker = end_marker = insert_tgt_markers
            
        elif isinstance(insert_tgt_markers, (list, tuple)) : 
            assert len(insert_tgt_markers) == 2, f"Provided quotation markers {insert_tgt_markers} must contain only two elements: start and end markers."
            st_marker = insert_tgt_markers[0]
            end_marker = insert_tgt_markers[1]


        return index, orig_text[context_start_byte:st_tgt_byte] + \
                    st_marker + \
                    orig_text[st_tgt_byte:e_tgt_byte] + \
                    end_marker + \
                    orig_text[e_tgt_byte:context_end_byte]


    def _process_one_book(self, book_id, num_tokens=100, insert_tgt_markers:bool=True, subset=None):
        """Helper function to get the context of all quotes in a single book."""        
        # files = glob.glob(os.path.join(self.libriquote_path, '*', book_id, '*', '*.json'))
        files = [self.recording2qpath[i] for i in self.bid2fs[book_id]]
        split = files[0].split('/')[-4]

        with open(os.path.join(self.libriquote_path, split, book_id, 'clean_text.txt')) as f: 
            orig_text = f.read()

        all_data = []
        for idx, fl in enumerate(files): 
            with open(fl) as f : 
                l = self._get_ordered_list(json.load(f))
                all_data.extend([(idx, ll) for ll in l])

        quotations = [idx for idx in range(len(all_data)) if all_data[idx][1]['is_quote']]
        out_q = {k:[] for k in files}
        
        for qidx in quotations : 
            _, context = self._get_context_one_quote(qidx, all_data, orig_text, num_tokens, insert_tgt_markers)
            filename = files[all_data[qidx][0]]
            data = all_data[qidx][1]
            data['context'] = context
            out_q[filename].append(data)
        return book_id, out_q
    
    def _process_one_recording(self, recording_name, num_tokens=100, insert_tgt_markers:bool=True, subset=None):
        """Helper function to get the context of all quotes in a single recording."""        

        path = self.recording2qpath[recording_name]
        book_path = '/'.join(path.split('/')[:-2])
        book_id = path.split('/')[-3]
        with open(os.path.join(book_path, 'clean_text.txt')) as f: 
            orig_text = f.read()
        with open(path) as f :  
            all_data = self._get_ordered_list(json.load(f))
            all_data = [(0, ll) for ll in all_data]
            
        quotations = [idx for idx in range(len(all_data)) if all_data[idx][1]['is_quote']]
        if subset is not None :
            quotations = [quotations[i] for i in subset]
            
        out_q = {path: []}
        
        for qidx in quotations : 
            _, context = self._get_context_one_quote(qidx, all_data, orig_text, num_tokens, insert_tgt_markers)
            # filename = path
            data = all_data[qidx][1]
            data['context'] = context
            out_q[path].append(data)
        return book_id, out_q
    
    def __call__(self, book_ids:List[str]=None, split:str = None, num_tokens=100, insert_tgt_markers:Union[bool, str, List[str], Tuple[str]]=True) :
        """Retrieves contextual information for each quote in the provided `book_ids`, or for each book in the provided LibriQuote `split` in parallel. Note that you can find `book_ids` with the `list_libriquote_ids` method.
        The segmentation uses paragraphs (as they come in the original text) as boundaries, meaning that each left and right context will accumulate N paragraphs until reaching the maximum number of tokens allowed.
        
        arguments:
        `book_ids`: a List of strings matching book identifiers in LibriQuote.
        `split`: a string indicating the LibriQuote split. If `book_ids` is also set, will only process those `book_ids` that comes in the `split`.
        `num_tokens`: int, number of maximum tokens allowed independently in left and right context.
        `insert_tgt_markers`: bool, str or List[str], if set to True will insert default markers at the beginning and end of the target quote in context. If set to a string, will insert this string insted. If a list or tuple of 2 elements, will use the first element as beginning marker and second element as ending marker.        
        
        returns:
        `A dictionnary {`book_id`: {`recording_name` : quote_info_with_context_dict, ...}, ...}
        """
        if isinstance(book_ids, str) : 
            book_ids = [book_ids]
        if split == 'benchmark' : 
            raise ValueError('Context is already provided in the benchmark data. If you want to modify the already attributed context, please consider creating context for the right quotes in the `test` set.')
        
        subset = None
        if split is not None : 
            assert split in ['train', 'train_filt', 'dev', 'test'], "`split` argument must be one of LibriQuote splits."
            if book_ids is not None : 
                # getting the the right recordings that matches both split and book
                assert isinstance(book_ids, list), "`book_ids` argument must be a list of LibriQuote book identifiers. You can use the `list_libriquote_ids` method to find them."
                split_recordings = set(self.s2fs[split])
                book_recordings =  set(sum([self.bid2fs[i] for i in book_ids],[]))
                processing_ids = split_recordings & book_recordings
                func = self._process_one_recording
                
                if len(processing_ids) == 0 : 
                    raise ValueError(f"Could not find requested book_ids in the following split: {split}")
            else : 
                processing_ids = self.s2fs[split]
                func = self._process_one_recording
                if split == 'train_filt' : 
                    subset = self.train_filtered_ids
        else : 
            assert isinstance(book_ids, list), "`book_ids` argument must be a list of LibriQuote book identifiers. You can use the `list_libriquote_ids` method to find them."
            processing_ids = book_ids
            func = self._process_one_book
            
        if not subset :
            subset={k : None for k in processing_ids}
        
        
        out = {}
        for i in tqdm(processing_ids) :
            res = func(
                i,
                num_tokens=num_tokens,
                insert_tgt_markers=insert_tgt_markers,
                subset=subset[i])
            out[res[0]] = res[1]
            
        return out 
        
    def process_recording(self, recording_name:str, num_tokens:int=100, insert_tgt_markers:Union[bool, str, List[str], Tuple[str]]=True, subset:List[int]=None) : 
        """Retrieves contextual information for quotes in the provided recording. You can find recording names with the `list_libriquote_recordings` method.
        The segmentation uses paragraphs (as they come in the original text) as boundaries, meaning that each left and right context will accumulate N paragraphs until reaching the maximum number of tokens allowed.
        
        arguments:
        `recording_name`: a string matching a LibriQuote recording.
        `num_tokens`: int, number of maximum tokens allowed independently in left and right context.
        `insert_tgt_markers`: bool, str or List[str], if set to True will insert default markers at the beginning and end of the target quote in context. If set to a string, will insert this string insted. If a list or tuple of 2 elements, will use the first element as beginning marker and second element as ending marker.
        
        returns:
        `A tuple (`book_id`, recording_data_with_context}
        """
        return self._process_one_recording(
            recording_name,
            num_tokens=num_tokens,
            insert_tgt_markers=insert_tgt_markers,
            subset=subset
        )