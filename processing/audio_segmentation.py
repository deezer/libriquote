import glob, os, json
import soundfile as sf
import numpy as np 
from typing import List, Tuple, Optional, Union
from multiprocessing import Pool
import multiprocessing as mp
from tqdm.auto import tqdm 
from processing.text_utils import LibriQuoteUtils
from functools import partial

class AudioSegmenter(LibriQuoteUtils): 
    def __init__(
        self,
        librilight_path: str,
        libriquote_path: str
    ) :
        super().__init__(libriquote_path)
        """Main class for segmenting LibriQuote audios. We expect users to have downloaded LibriLight audio recordings, and will use them as the main audio files.
        Please note that segmenting the full train dataset is long, and might be causing OOM issues when loading everything in memory.
        A simple solution would be to process books iteratively and save each audio required audio segments. Note that saving all audio segments might end up in using around 1TB of disk space.
        Alternatively, the `read_segment`method can be used to load a specific (`book_id`, `quote_id`) segment.
        arguments:
        `librilight_path` path to LibriLight split folders
        `libriquote_path` path to LibriQuote split folders
        """
        self.librilight_path = librilight_path
        librilight_audiof =  glob.glob(os.path.join(self.librilight_path, '*', '*', '*', '*.flac'))
        self.recording2lpath = {}
        valid_ids = self.librilight_name_to_bookid()
        for k in librilight_audiof : 
            rest, recording = os.path.split(k)
            librilight_name = os.path.split(rest)[-1]
            if librilight_name in valid_ids :
                self.recording2lpath[valid_ids[librilight_name] + '.' + recording.replace('.flac','')] = k
                
        
    def get_audio_book_info(self, libriquote_id) : 
        """Given a LibriQuote ID, returns all available recordings and the following metadatas:
        - number of quotation segments
        - number of narration segments
        - LibriQuote split it belongs to
        - Speaker ID
        - Full path to the alignment file
        - Full path to the associated LibriLight audio file."""
        
        data = self.get_book_info(libriquote_id)
        out = {}
        for k,v in data.items() : 
            if libriquote_id + '.' + k in self.recording2lpath : 
                audio_path = self.recording2lpath[libriquote_id + '.' + k]
            else : 
                audio_path = None
            v['audio_path'] = audio_path
            out[k] = v
        return out
    
    def list_available_librilight_recordings(self) :
        out={}
        for k,v in self.recording2lpath.items() : 
            vv = v.split('/')[-2]
            if vv not in out : 
                out[vv] = []
            out[vv].append(k)
        return out
    
    def _segment_audio(
        self,
        audio,
        start_s,
        end_s,
        sr
    ) : 
        if len(audio) == 1 : 
            return audio[:, int(np.floor(start_s * sr)) : int(np.ceil(end_s * sr))]
        else : 
            return audio[int(np.floor(start_s * sr)) : int(np.ceil(end_s * sr))]

    def _routine_input_tests(self, libriquote_ids, librilight_ids, split) : 
        
        if all(
            [libriquote_ids is None,
             librilight_ids is None,
             split is None]
        ) : 
            raise ValueError("Please provide one of the following arguments: `libriquote_ids`, `librivox_ids`, `split`.")

        if all([libriquote_ids is not None, librilight_ids is not None]) : 
            raise ValueError("Please provide only one of `libriquote_ids` or `librivox_ids`")

        if libriquote_ids is not None : 
            assert isinstance(libriquote_ids, list), "Provided ids must be a list with string elements `book_id`. Check the `list_libriquote_ids` method."
            assert len(libriquote_ids) > 0, "Please provide at least one book id. Check the `list_libriquote_ids` method." 
            assert isinstance(libriquote_ids[0], str), "Provided ids must be a list with string elements `book_id`. Check the `list_libriquote_ids` method."
            return 0
        if librilight_ids is not None : 
            assert isinstance(librilight_ids, list), "Provided ids must be a list with elements `librilight_id`. Check the `list_librilight_ids` method."
            assert len(librilight_ids) > 0, "Please provide at least one element `librilight_id`. Check the `list_librilight_ids` method." 
            return 1
        if split is not None : 
            assert isinstance(split, str), "Provided split must be either a LibriLight split (`small`, `medium`, `large`) or a LibriQuote split (`train`, `dev`, `test`)"
            assert split in ['small', 'medium', 'large', 'train', 'train_filt', 'test', 'dev', 'benchmark'], "Provided split must be either a LibriLight split (`small`, `medium`, `large`) or a LibriQuote split (`train`, `dev`, `test`)"
            return 2 

    def _segment_one(self, process_list:List[Tuple[str,str,List[int]]], load_audio=True, segment_type:str='cut', q_durs=None, n_durs=None) : 
        """
        Processes a list of `(align_file, audio_file)` to load each audio segments provided in the `align_file`.
        """
        if segment_type == 'cut' : 
            skey, ekey = 'cut_start_time', 'cut_end_time'
        else : 
            skey, ekey = 'start_time', 'end_time'

        out = {}
        qpath, lpath, subset = process_list

        if load_audio : 
            audio, sr = sf.read(lpath)
        book_id = qpath.split('/')[-3]
        if book_id not in out :
            out[book_id] = {}
            
        with open(qpath) as f : 
            data = json.load(f)
            data2 = data.copy()
            if subset is not None : 
                q_range = len(subset)
                data2['quotations'] = [data2['quotations'][i] for i in subset]
            else : 
                q_range = len(data['quotations'])

            if load_audio: 
                
                for idx in range(q_range) : 
                    q = data2['quotations'][idx]
                    if q_durs is not None : 
                        if any([
                (q['cut_end_time'] - q['cut_start_time'] ) <= q_durs[0],
                (q['cut_end_time'] - q['cut_start_time'] ) >= q_durs[1],
                        ]) :
                            continue

                    segment = self._segment_audio(audio, q[skey], q[ekey], sr)
                    data2['quotations'][idx]['audio'] = segment
                    data2['quotations'][idx]['sr'] = sr
                data2['quotations'] = [i for i in data2['quotations'] if 'audio' in i]
                
                for idx in range(len(data['narrations'])) : 
                    q = data['narrations'][idx]
                    if n_durs is not None : 
                        if any([
                (q['cut_end_time'] - q['cut_start_time'] ) <= n_durs[0],
                (q['cut_end_time'] - q['cut_start_time'] ) >= n_durs[1],
                        ]) :
                            continue
                    segment = self._segment_audio(audio, q[skey], q[ekey], sr)
                    data2['narrations'][idx]['audio'] = segment
                    data2['narrations'][idx]['sr'] = sr
                data2['narrations'] = [i for i in data2['narrations'] if 'audio' in i]

            out[book_id][qpath] = data2
        return out 

    def _process_all(self, recordings, load_audio, segment_type, subset, q_durs, n_durs) : 

        to_process = []
        total = len(recordings)
        cnt = 0 
        for rec in recordings: 
            if rec in self.recording2lpath : 
                data = (self.recording2qpath[rec], self.recording2lpath[rec])
                if subset is not None :
                    data += (subset[rec],)
                else :
                    data += (None,)
                to_process.append(data) 
                cnt +=1
        print(f'Found {cnt}/{total} recording files in LibriLight path.')

        
        out = {}
        for p in tqdm(to_process) : 
            res = self._segment_one(p, load_audio, segment_type, q_durs, n_durs)
            for k,v in res.items() : 
                if k not in out : 
                    out[k] =v 
                else : 
                    out[k].update(v)
        return out
    
    def __call__(
        self,
        split: Optional[Union[str, None]] = None,
        libriquote_ids: Optional[Union[str, List[Tuple[str, str]], None]]  = None,
        librilight_ids: Optional[Union[str, List[str], None]]  = None,
        load_audio:bool=True,
        quotation_dur_params:Union[List[float], Tuple[float,float]]=None,
        narration_dur_params:Union[List[float], Tuple[float,float]]=None,
        segment_type: str = 'cut',
    ) : 
        """Performs audio segmentation on specific books or specific LibriQuote splits using LibriLight recordings. Note that `split`, `libriquote_ids` and `librilight_ids` arguments are mutually exclusive.
        arguments:
        `split` one of LibriQuote splits 'train', 'train_filt', 'dev', 'test. Using this argument will process the full split.
        `libriquote_ids` a list of 'book_id' strings to be processed. You can check ids with the `list_libriquote_ids()` method.
        `librilight_ids` a list of 'librilight_ids' strings to be processed. You can check names with the `list_librilight_ids()` method.
        `load_audio` whether to load audio or not. Faster when not loading audio.
        `quotation_dur_params` (minimum, maximum) duration in seconds to consider a quotation segment
        `narration_dur_params` (minimum, maximum) duration in seconds to consider a nararation segment
        `segment_type` whether to use the VAD cutted segments or the normal segments. We recommend putting to `cut`.
        Returns:
        a dictionnary {`split`: {`book_id` : book_alignments_and_metadata}}
        """
        
        if isinstance(libriquote_ids, str) : 
            libriquote_ids = [libriquote_ids]
        if isinstance(librilight_ids, str) : 
            librilight_ids = [librilight_ids]
        
        process_mode = self._routine_input_tests(libriquote_ids, librilight_ids, split)
        subset = None
        if process_mode == 2 : 
            # split mode 
            if split  == 'train_filt' : 
                subset = self.train_filtered_ids
            elif split == 'benchmark' : 
                subset = self.benchmark_ids
                
            id_list = self.list_libriquote_recordings(split)

        elif process_mode == 1 : 
            id_list = sum([self.librilight_name_to_recordings(i) for i in librilight_ids], [])
        else :
            id_list = sum([self.id_to_recordings(i) for i in libriquote_ids], [])

        return self._process_all(
            id_list,
            load_audio=load_audio,
            segment_type=segment_type,
            subset=subset,
            q_durs=quotation_dur_params,
            n_durs=narration_dur_params)
        
    def read_recording(
        self,
        recording_name:str,
        segment_type:str='cut',
        quotation_subset:List[int]=None,
        quotation_dur_params:Union[List[float], Tuple[float,float]]=None,
        narration_dur_params:Union[List[float], Tuple[float,float]]=None
        ) : 
        """
        Main function to load a all segments from a specific recording. You can find recording names using the `list_libriquote_recordings` method.

        arguments:
        `recording_name` name of a single recording such as `marylouiseinthecountry_18_baum_64kb`.
        `segment_type` whether to use the VAD cutted segments or the normal segments. We recommend putting to `cut`.
        `quotation_subset`: List of quotations indices to keep in the recording, in case only some are desirable.
        `quotation_dur_params`: Tuple/List of floats, range (minimum, maximum) duration in seconds of quotations to be considered. Those not falling in the range will be discarded.
        `narration_dur_params`: Tuple/List of floats, range (minimum, maximum) duration in seconds of narrations to be considered. Those not falling in the range will be discarded.

        returns:
        loaded data
        """
        book_id, rec, data = self.load_recording(recording_name)
        
        audio_file = glob.glob(os.path.join(self.librilight_path, '*', data['librivox_src']))
        if len(audio_file) == 0 : 
            raise ValueError(f'Could not find recording `{recording_name}` in LibriLight files.')
        else : 
            audio_file = audio_file[0]
            
        if segment_type == 'cut' : 
            skey, ekey = 'cut_start_time', 'cut_end_time'
        else : 
            skey, ekey = 'start_time', 'end_time'
        
        data2 = data.copy()
        audio, sr = sf.read(audio_file)
        if quotation_subset is not None : 
            data2['quotations'] = [data['quotations'][idx] for idx in quotation_subset]
            
        for idx in range(len(data2['quotations'])) : 
            q = data2['quotations'][idx]
            if quotation_dur_params is not None : 
                if any([
        (q['cut_end_time'] - q['cut_start_time'] ) <= quotation_dur_params[0],
        (q['cut_end_time'] - q['cut_start_time'] ) >= quotation_dur_params[1],
                ]) :
                    continue
                
            segment = self._segment_audio(audio, q[skey], q[ekey], sr)
            data2['quotations'][idx]['audio'] = segment
            data2['quotations'][idx]['sr'] = sr
        data2['quotations'] = [i for i in data2['quotations'] if 'audio' in i]

        for idx in range(len(data['narrations'])) : 
            q = data['narrations'][idx]
            if narration_dur_params is not None : 
                if any([
        (q['cut_end_time'] - q['cut_start_time'] ) <= narration_dur_params[0],
        (q['cut_end_time'] - q['cut_start_time'] ) >= narration_dur_params[1],
                ]) :
                    continue
            segment = self._segment_audio(audio, q[skey], q[ekey], sr)
            data2['narrations'][idx]['audio'] = segment
            data2['narrations'][idx]['sr'] = sr
        data2['narrations'] = [i for i in data2['narrations'] if 'audio' in i]

        return data2
    
    def read_segment(
        self,
        recording_name:str,
        index:int,
        narr_type:str='quotations',
        segment_type:str='cut') : 
        """
        Main function to load a single segment from a specific recording. You can find recording names using the `list_libriquote_recordings` method, and more information such as number of quotations/narrations with the `get_audio_book_info` method.
        Useful to inspect a single segment.
        
        arguments:
        `recording_name` name of a single recording such as `marylouiseinthecountry_18_baum_64kb`.
        `index: int` index of the segment you wish to load
        `narr_type` one of `quotations` or `narrations` depending on if you want to load a narration or quotation segment
        `segment_type` whether to use the VAD cutted segments or the normal segments. We recommend putting to `cut`.

        returns:
        a tuple `(audio, sampling_rate, loaded_text_data)`.
        """
        if segment_type == 'cut' : 
            skey, ekey = 'cut_start_time', 'cut_end_time'
        else : 
            skey, ekey = 'start_time', 'end_time'
        
        book_id, rec, data = self.load_recording(recording_name)
        audio_file = glob.glob(os.path.join(self.librilight_path, '*', data['librivox_src']))
        if len(audio_file) == 0 : 
            raise ValueError(f'Could not find recording `{recording_name}` in LibriLight files.')
        else : 
            audio_file = audio_file[0]

        s, e = data[narr_type][index][skey], data[narr_type][index][ekey]
        s,e = int(np.floor(s * 16000)), int(np.ceil(e * 16000))
        a, sr = sf.read(audio_file, start=s, stop=e)
        return a, sr, data[narr_type][index]