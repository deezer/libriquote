import json, os, glob 
import re 
from tqdm.auto import tqdm 
from typing import List 

class LibriQuoteUtils: 
    def __init__(
        self,
        libriquote_path: str
    ) : 
        self.libriquote_path = libriquote_path
        
        self.s2fs, self.q2l, self.l2q, self.q2s = {'train' : [], 'dev' : [], 'test' : []}, {}, {}, {}
        self.recording2qpath, self.bid2fs = {}, {}
        
        for split in ['train', 'dev', 'test'] : 
            curr = json.load(open(os.path.join(self.libriquote_path, f'{split}_id_mapper.json')))
            # split --> book_ids
            self.q2l[split] = list(curr.keys())
            for k in curr : 
                if k not in self.bid2fs : 
                    self.bid2fs[k] = []
                for f in curr[k]['files'] : 
                    self.q2s[k + '.' +os.path.split(f)[-1].replace('.json','')] = split
                    self.s2fs[split].append(k + '.' + os.path.split(f)[-1].replace('.json',''))
                    self.recording2qpath[k + '.' +os.path.split(f)[-1].replace('.json','')] = os.path.join(self.libriquote_path, f)
                    self.bid2fs[k].append(k + '.' + os.path.split(f)[-1].replace('.json',''))
            # librivox_id --> book_id
            self.l2q.update({v['librilight_name']:k for k,v in curr.items()})
        self.train_filtered_ids = json.load(open(os.path.join(self.libriquote_path, 'train_filtered_ids.json')))
        self.s2fs['train_filt'] = list(self.train_filtered_ids)
        
        self.benchmark_ids = json.load(open(os.path.join(self.libriquote_path, 'benchmark_test_ids.json')))
        self.s2fs['benchmark'] = list(self.benchmark_ids)
        
        self.recording_list = self.list_libriquote_recordings()

    def id_to_recordings(self, book_id:str=None):
        if not book_id : 
            return self.bid2fs
        else : 
            assert book_id in self.bid2fs, f"Book ID {book_id} could not be found. You can list book ids with the `list_libriquote_ids` method."
            return self.bid2fs[book_id]
    
    
    def get_split_names(self) :
        return ['train', 'train_filt', 'dev', 'test', 'benchmark']
    
    def recording_to_id(self, recording_name:str=None):
        if not recording_name : 
            return {r: v.split('/')[-3] for r,v in self.recording2qpath.items()}
        else : 
            assert recording_name in self.recording2qpath, f"Recording {recording_name} could not be found. You can list recording names with the `list_libriquote_recordings` method."
            return self.recording2qpath[recording_name].split('/')[-3]
        
    def librilight_name_to_bookid(self, librilight_name:str=None): 
        if not librilight_name : 
            return self.l2q
        else : 
            assert librilight_name in self.l2q, f"LibriLight name {librilight_name} could not be found."
            return self.l2q[librilight_name]
    
    def id_to_librilight_name(self, book_id:str=None): 
        if not book_id : 
            if not hasattr(self, "il2q") : 
                self.il2q = {v:k for k,v in self.l2q.items()}
            return self.il2q
        else : 
            assert book_id in self.l2q.values(), f"Book ID {book_id} could not be found. You can list book ids with the `list_libriquote_ids` method."
            if not hasattr(self, "il2q") : 
                self.il2q = {v:k for k,v in self.l2q.items()}
            return self.il2q[book_id]
    
    def list_libriquote_ids(self, split = None) : 
        """Lists all libriquote ids along with their split."""
        if split is None :
            return sum([[(split, k) for k in self.q2l[split]] for split in ['train', 'dev', 'test']], [])
        else :
            return [k for k in self.q2l[split]]

    def list_libriquote_recordings(self, split = None) : 
        """Lists all libriquote ids along with their split."""
        if split is None : 
            return sum([[(split, k) for k in self.s2fs[split]] for split in ['train', 'dev', 'test']], [])
        else :
            return self.s2fs[split]

    def librilight_name_to_recordings(self, librilight_name=None) : 
        full = {k : self.id_to_recordings(v) for k,v in self.librilight_name_to_bookid().items()}
        
        if librilight_name : 
            return full[librilight_name]
        else : 
            return full
        
    def get_book_info(self, libriquote_id) : 
        """Given a LibriQuote ID, returns all available recordings and the following metadatas:
        - number of quotation segments
        - number of narration segments
        - LibriQuote split it belongs to
        - Speaker ID
        - Full path to the alignment file
        - Full path to the associated LibriLight audio file."""
        assert libriquote_id in self.l2q.values(), f"Could not find libriquote_id `{libriquote_id}` in LibriQuote books."
        
        files =  glob.glob(os.path.join(self.libriquote_path, '*', libriquote_id, '*', '*.json'))
        out = {}
        for fl in files : 
            with open(fl) as f : 
                data = json.load(f)
                out[fl.split('/')[-1].replace('.json','')] = {
                    'num_quotations' : len(data['quotations']),
                    'num_narrations' : len(data['narrations']),
                    "split" : fl.split('/')[-4],
                    "speaker" : fl.split('/')[-2],
                    'align_path' : fl,
                }
        return out

    def __call__(self) : 
        raise NotImplementedError

    def load_recording(self, recording_name:str = None) : 
        """Loads all segments belonging to a single `recording_name`. You can find recording names with `list_libriquote_recordings` method.
        
        arguments:
        `recording_name`: str, name of a chapter recording.

        returns:
        A dictionary `segments`
        """
        if '.json' in recording_name : 
            recording_name = recording_name.replace('.json', '')
            
        assert recording_name in self.recording2qpath, f"Could not find {recording_name} in LibriQuote recordings. Check out the `list_libriquote_recordings` to find recording names."
        
        path = self.recording2qpath[recording_name]
        book_id = path.split('/')[-3]
        with open(path) as f : 
            return book_id, recording_name, json.load(f)
    
    def load_book(self, libriquote_id:str = None) : 
        """Loads either all segments belonging to all recordings of a `libriquote_id`. You can find all libriquote IDs with the `list_libriquote_ids` method.
        
        arguments:
        `libriquote_id`: str, libriquote ID of a book

        returns:
        A dictionary `{'libriquote_id' : {'recording_name' : segments, ...}, ...}`
        """
        assert libriquote_id in self.bid2fs, f"Could not find {libriquote_id} in LibriQuote book ids. Check out the `list_libriquote_ids` method to find all book ids."
        recordings = self.bid2fs[libriquote_id]
        
        out = {libriquote_id:{}}
        for rec in recordings :
            _, _, data = self.load_recording(rec)
            
            out[libriquote_id][self.recording2qpath[rec]] = data
        return out

    def load_split(self, split:str):
        """Loads all segments for a particular LibriQuote split."""
        assert split in ['train', 'train_filt', 'dev', 'test', 'benchmark'], "Please input one of LibriQuote splits."
        out = {}
        if split in ['train', 'dev', 'test'] : 
            recordings = self.s2fs[split]
            
            for f in tqdm(recordings) : 
                book_id, rec, data = self.load_recording(f)
                if book_id not in out  :
                    out[book_id] = {}
                out[book_id][self.recording2qpath[rec]] = (data)
                
        elif split == 'train_filt' : 
            recordings = self.train_filtered_ids

            for f, v in tqdm(recordings.items()) : 
                book_id, rec, data = self.load_recording(f)
                if book_id not in out  :
                    out[book_id] = {}
                try : 
                    data['quotations'] = [data['quotations'][i] for i in v]
                except : 
                    print(book_id, rec)
                    raise ValueError('tt')
                out[book_id][self.recording2qpath[rec]] = (data)
                
        elif split =='benchmark' : 
            recordings = self.s2fs['test']
            
            for r in tqdm(recordings) : 
                path = self.recording2qpath[r].replace('test/', 'benchmark/') 
                book_id = path.split('/')[-3]
                with open(path) as f : 
                    data = json.load(f)
                if book_id not in out : 
                    out[book_id] = {}
                out[book_id][path] = (data)
        return out
        
def is_quote_punct(string) : 
    return string in ["“", "”", '"',  "“", "‘", "’", "'"]

def _segment_into_paragraph(text, double_blank=True):
    if double_blank:
        pattern = "\n\s*\n\s*"
    else:
        pattern = "\n\s*"
    indices = [
        {"break_st": m.start(), "break_end": m.end() - 1}
        for m in re.finditer(pattern, text)
    ]
    paragraphs = []
    if len(indices) == 0:
        indices = [
            {
                "text_st": 0,
                "text_end": len(text) - 1,
                "break_st": len(text) - 1,
                "break_end": len(text) - 1,
            }
        ]
        paragraphs.append(text)
    else:
        for i in range(len(indices)):
            if i == 0:
                indices[i]["text_st"] = 0
            else:
                indices[i]["text_st"] = indices[i - 1]["break_end"] + 1
            indices[i]["text_end"] = indices[i]["break_st"]
            paragraphs.append(text[indices[i]["text_st"] : indices[i]["break_end"] + 1])

    if indices[-1]["break_end"] != len(text) - 1:
        indices.append(
            {
                "text_st": indices[-1]["break_end"] + 1,
                "text_end": len(text) - 1,
                "break_st": len(text) - 1,
                "break_end": len(text) - 1,
            }
        )
        paragraphs.append(text[indices[-1]["text_st"] : indices[-1]["break_end"] + 1])

    for i in range(len(indices)):
        indices[i] = {"st": indices[i]["text_st"], "end": indices[i]["text_end"]}

    return paragraphs, indices


def _should_verify_dur(min_dur, max_dur) : 
    return any([
            isinstance(min_dur, (float,int)),
            isinstance(max_dur, (float,int))
        ])
def _get_timed_candidates(candidates, min_dur, max_dur) : 
    return [
        i for i in candidates if all(
            [
                (i[1]['cut_end_time'] - i[1]['cut_start_time'] ) <= max_dur,
                (i[1]['cut_end_time'] - i[1]['cut_start_time'] ) >= min_dur,
            ]
        )
    ]