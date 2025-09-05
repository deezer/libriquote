# LibriQuote Splits

LibriQuote comes with 5 splits:
- The `train` split, that contains 2991 unique books and 64408 recordings, featuring 5359 hours of quotation and 12723 hours of narration segments.
- The `train_filt` split, that contains 2900 unique books across 50605 recordings. 
It features 379 hours of quotations, and each quotation was selected based on expressive narrative information.
- The `dev` split used for testing models. The speaker overlaps with the training speakers, so it can't be used to evalute zero-shot TTS.
- The `test` split contains 27 unique books with unseen speakers. It is different from the `benchmark` split (see below) as it contains all quotations and all narrations for each book. It is primilary inteded for researchers that want to derive a `benchmark` with different attributes (i.e different `context` or different `reference` speech sample or different `target` speech sample).
- The `benchmark` split it the split used for benchmarking. It comes with its own set of `reference` speech sample (narration segment) and `context`. All audios are available in the [Huggingface repository](https://huggingface.co/datasets/gasmichel/LibriQuote/tree/main/test_audios).

# Processing LibriQuote

This repository contains the necessary Python code to navigate and to process LibriQuote data.


The repository contains mainly 3 python tools:
- A [`basic tool`](#navigating-libriquote) that helps navigating libriquote files.
- An [`audio segmentation`](#audio-segmentation) tool that processes LibriQuote alignments to segment LibriLight audio files and provide the actual audio data.
- A [`context retrieval`](#context-retrieval) tool that links quotations with the book context in which they occur.
- A [`narration matcher`](#narration-matching) tool that matches LibriQuote quotations with narration segments.

## LibriLight dependancy

We assume that you already have downloaded LibriLight audio files, as our files will follow -- to some extent -- LibriLight format. If you haven't downloaded LibriLight files, you can do so [here](https://github.com/facebookresearch/libri-light/blob/main/data_preparation/README.md).

- We can not provide download links for LibriQuote audio files yet (but hope to do so in a near future), so LibriLight audio files are necessary for now.

- For users who would only wants LibriLight audio files associated with LibriQuote data, we provide a [bash script](../librilight_matching/) that will `untar` only the necessary audio files.

## LibriQuote Dataset

The LibriQuote dataset is accessible on [HuggingFace](https://huggingface.co/datasets/gasmichel/LibriQuote). You can clone the dataset with `git`:

```
git clone https://huggingface.co/datasets/gasmichel/LibriQuote
```
For other cloning options, see [Huggingface documentation](https://huggingface.co/docs/hub/repositories-getting-started#cloning-repositories).

## Installation

First, install the required python dependancies. You can do so with the following:

```
python3 -m venv /tmp/libriquote_process/
source /tmp/libriquote_process/bin/activate
pip install -r requirements.txt
```

## Navigating LibriQuote
We provide some examples below of how to navigate LibriQuote with the basic `LibriQuoteUtils` class. Note that subsequent tools inherit from `LibriQuoteUtils` so they will have access to the same methods for navigation.

```python
from processing.text_utils import LibriQuoteUtils

helper = LibriQuoteUtils('PATH_TO_LIBRIQUOTE_FILES')

# List LibriQuote book IDs and their associated split
helper.list_libriquote_ids()
# [('train', '6868'),
#  ('train', '4800'),
#  ('train', '15206'),
#  ...]

# List LibriQuote recordings and their associated split
helper.list_libriquote_recordings()
# [('train', 'prairie_28_cooper_64kb'),
#  ('train', 'prairie_10_cooper_64kb'),
#  ('train', 'prairie_32_cooper_64kb'),
#  ...]

# List all recordings associated with a book ID
helper.id_to_recordings()
# {'6868': ['prairie_28_cooper_64kb',
#   'prairie_10_cooper_64kb',
#   'prairie_32_cooper_64kb',
#  ...]

# Mapping between book IDs and LibriLight recording names
helper.id_to_librilight_name()
# {'6868': 'prairie_wp_librivox_64kb_mp3',
#  '4800': 'robin_hood_1101_librivox_64kb_mp3',
#  '15206': 'onethingneedful_2011_librivox_64kb_mp3'
#  ...]

# Show some book information
helper.get_book_info('6868')
# {'prairie_28_cooper_64kb': {'num_quotations': 43,
#   'num_narrations': 45,
#   'split': 'train',
#   'speaker': '4441',
#   'align_path': '/data/LibriQuote/train/6868/4441/prairie_28_cooper_64kb.json',
# 'prairie_10_cooper_64kb' : {...}},

# Load libriquote files for single recording
book_id, rec_name, data = helper.load_recording('prairie_28_cooper_64kb')

# Get segment duration (in seconds)
helper.get_segment_duration(data['quotations'][0])
# 2.615000000000009
helper.get_segment_duration(data['narrations'][0])
# 1.5851249999999908

# Predicted Narrative Information for quotations
print(data['quotations'][0]['narrative_prediction'])
# {'returned': {'id': '1', 'type': 'verb', 'confidence': 10},
#  'calmly': {'id': '1', 'type': 'adverb', 'confidence': 10}}

# Load libriquote files for all recordings belonging to a book id
data = helper.load_book('6868')

# Load all files belonging to a LibriQuote split (train, train_filt, dev, test)
helper.load_split('dev')
print(len(data['10682']['/data/LibriQuote/dev/10682/10159/mysteriesoflondon2_062_reynolds_64kb.json']['quotations']))
# 63
print(len(data['10682']['/data/LibriQuote/dev/10682/10159/mysteriesoflondon2_062_reynolds_64kb.json']['narrations']))
# 56

# Load the benchmark data. The structure is a bit different here, as narrations and context have already been matched for each quotations.
print(data['132']['/data/LibriQuote/benchmark/132/1320/lastofthemohicans_21_cooper_64kb.json']['quotations'][0])
# {'text': '“When I found that the home path of the Hurons run north,',
#  'start_byte': 508335,
#  'end_byte': 508392,
#  'is_quote': True,
#  'cut_start_time': 175.92499572753906,
#  'cut_end_time': 178.87005822753906,
#  'narration': {'text': ' said Uncas, pointing north and south, at the evident marks of the broad trail on either side of him,',
#   'cut_start_time': 300.1750122070313,
#   'cut_end_time': 306.09007470703125,
#   'audio_path': 'prompt-wavs/132.lastofthemohicans_21_cooper_64kb_0.flac'},
#  'context': 'After proceeding a few miles, the progress of Hawkeye, who led the advance, became more deliberate and watchful. He often stopped to examine the trees; nor did he cross a rivulet without attentively considering the quantity, the velocity, and the color of its waters. Distrusting his own judgment, his appeals to the opinion of Chingachgook were frequent and earnest. During one of these conferences Heyward observed that Uncas stood a patient and silent, though, as he imagined, an interested listener. He was strongly tempted to address the young chief, and demand his opinion of their progress; but the calm and dignified demeanor of the native induced him to believe, that, like himself, the other was wholly dependent on the sagacity and intelligence of the seniors of the party. At last the scout spoke in English, and at once explained the embarrassment of their situation.\n\n<|quote_start|>“When I found that the home path of the Hurons run north,”<|quote_end|> he said, “it did not need the judgment of many long years to tell that they would follow the valleys, and keep atween the waters of the Hudson and the Horican, until they might strike the springs of the Canada streams, which would lead them into the heart of the country of the Frenchers. Yet here are we, within a short range of the Scaroons, and not a sign of a trail have we crossed! Human natur’ is weak, and it is possible we may not have taken the proper scent.”',
#  'narrative_information_pred': {'said': {'id': '1',
#    'type': 'verb',
#    'confidence': 10}},
#  'audio_path': 'wavs/132.lastofthemohicans_21_cooper_64kb_0.flac',
#  'original_index': 0}

# Show the quotation subsets for the splits `train_filt` or `benchmark`
print(helper.list_quotation_subset('benchmark'))
# {'6773.thousand_nights_vol07_29_burton_64kb': [1,
#   2,
#   3,
#   4,
# ...}
```

## Audio Segmentation

We provide examples of how to retrieve audio segments below. Note that this class requires to have saved necessary LibriLight audio files beforehand.

```python
from processing.audio_segmentation import AudioSegmenter

segmenter = AudioSegmenter('PATH_TO_LIBRILIGHT', 'PATH_TO_LIBRIQUOTE')

# Show available LibirLight audio files
segmenter.list_available_librilight_recordings()
# {'badge_courage_librivox_64kb_mp3': ['626.redbadgeofcourage_18_crane_64kb',
#   '626.redbadgeofcourage_17_crane_64kb',
#   '626.redbadgeofcourage_19_crane_64kb',
#   ...}

# Same as `get_book_info` but additionnaly returns 
# the path to the audio file if found in LibriLight path
helper.get_audio_book_info('626')
# {'redbadgeofcourage_18_crane_64kb': {'num_quotations': 25,
#   'num_narrations': 41,
#   'split': 'train',
#   'speaker': '816',
#   'align_path': '/data/LibriQuote/train/626/816/redbadgeofcourage_18_crane_64kb.json',
#   'audio_path': '/data/LibriLight/small/816/badge_courage_librivox_64kb_mp3/redbadgeofcourage_18_crane_64kb.flac'}
# ...},

# Load all audio segments from a book_id.
# Note that this will load both quotation and narration segments.
data = segmenter(libriquote_ids='6868')

# Load audio segments for multiple book_ids
data = segmenter(libriquote_ids=['6868','4800'])

# Load from LibriLight name
data = segmenter(librilight_ids='badge_courage_librivox_64kb_mp3')

# Load a full LibriQuote split
data = segment(split='train_filt')

# Put duration restrictions (seconds) on quotation duration or narration.
# This will only load segments that matches these restrictions
data = segmenter(
    libriquote_ids='6868',
    quotation_dur_params=(2,10),
    narration_dur_params=(1,30)
)

# Read segments for a single recording
data = segmenter.read_recording('626.redbadgeofcourage_18_crane_64kb')
print(len(data['quotations']))
# 25

# Single recording but only loads specific quotations
data = segmenter.read_recording(
    '626.redbadgeofcourage_18_crane_64kb', quotation_subset=[0,5])
print(len(data['quotations']))
# 2

# Read quotation/narration segment in a recording at index 0
audio, sr, segment_info = segmenter.read_segment(
    '626.redbadgeofcourage_18_crane_64kb',
    index=0,
    narr_type='quotations' # can also be `narrations`
)
np.allclose(audio , data['quotations'][0]['audio'])
# True
```

## Context Retrieval

In this example, we show how to retrieve varying contextual information for quotations, based on the original book text. Here, we use Spacy to tokenize the text, but the code can be easily adapted to use any tokenizer.

```python

from processing.context import ContextRetriever

retriever = ContextRetriever('PATH_TO_LIBRIQUOTE')

# Get context for all quotations in a recording
# By default, some markers will be added at the beginning and end of the quotation in the context.
book_id, data = retriever.process_recording(
    '6868.prairie_28_cooper_64kb',
    num_tokens=100, # Number of tokens before and after the quotation.
    insert_tgt_markers=True, # set to False to remove markers. Can also use personalized markers.
    )
print(data['/data/LibriQuote/train/6868/4441/prairie_28_cooper_64kb.json'][0]['text'])
# “they shall be given as plainly as you send them.”
print(data['/data/LibriQuote/train/6868/4441/prairie_28_cooper_64kb.json'][0]['context'])
#  calmly returned the trapper; <|Q|>“they shall be given as plainly as you send them.”<|Q|>
# “Friend!” repeated the squatter, eyeing the other for an instant, with an expression of indefinable meaning. “But it is no more than a word, and sounds break no bones, and survey no farms. Tell this thieving Sioux, then, that I come to claim the conditions of our solemn bargain, made at the foot of the rock.”
# When the trapper had rendered his meaning into the Sioux language, Mahtoree demanded, with an air of surprise — 

# Retrieving context only for a subset of quotations
book_id, data = retriever.process_recording(
    '6868.prairie_28_cooper_64kb',
    subset=[0,2])

# Full book based on book_id
data = retriever(book_ids='6868')

# Or multiple ids
data = retriever(book_ids=['6868', '626'])

# Or load a full split
data = retriever(split='train_filt')
```
## Narration Matching

This tool is used to match a quotation with a narration segment within the same recording. We used it to have a "neutral" (i.e narration) reference segment for each quotation in the test set.

```python
from processing.narration import NarrationToQuoteMatcher 

matcher = NarrationToQuoteMatcher('PATH_TO_LIBRIQUOTE')

# Matches for a single recording.
# By default, matches a quotation with its nearest narration segment in the book text.
book_id, data = matcher.process_one_recording(
    '6868.prairie_28_cooper_64kb',
    how='nearest', # can also be `random` for random matching in the same recording
    min_narr_dur=1, # minimum narartion duration in second to match
    max_narr_dur=20, # maximum narration duration in second to match
    subset= None, # similar as above, can be set to List[int] to only process specific quotations
    seed=None # Set to an integer for reproducible results when using `random`
)
print(data['/data/LibriQuote/train/6868/4441/prairie_28_cooper_64kb.json'][0]['narration'])
# {'text': ' calmly returned the trapper;',
#  'start_byte': 711827,
#  'end_byte': 711856,
#  'is_quote': False,
#  'start_time': 83.5999984741211,
#  'end_time': 85.76000213623047,
#  'cut_start_time': 83.8649984741211,
#  'cut_end_time': 85.45012347412109}

# process one or multiple books
out = matcher(
    book_ids = ['6868', '626'],
    matching_type='random',
    min_narr_dur=None,
    max_narr_dur=None,
    seed=2025
)

# or a full split
out = matcher(
    split = 'train_filt',
    matching_type='nearest',
    min_narr_dur=2,
    max_narr_dur=5
)
```