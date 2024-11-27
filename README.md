# Field Recordings Search Engine

This repository contains the codebase for a Field Recordings Search Engine, developed as part of my Master's thesis project. The goal of this project is to organize, process, and make field recordings easily searchable and accessible. The pipeline integrates advanced techniques for automatic source separation and speech recognition, enabling the extraction of meaningful information from audio recordings.

The project also includes automatic transcription indexing to structure the extracted text data and a fuzzy search engine that allows efficient retrieval of recordings based on approximate textual matches. Together, these components provide a comprehensive solution for managing and exploring large collections of field recordings.

---

## Features
- **Source Separation**: Separate vocals and instrumental sounds using the [ZFTurbo MDX model](https://github.com/ZFTurbo/Music-Source-Separation-Training).
- **Speech-to-Text**: Transcribe speech content in recordings using [OpenAI's Whisper](https://github.com/openai/whisper) for high-quality ASR.
- **Fuzzy Search Engine**: Index and search processed recordings using extracted metadata and textual content.

---

## Requirements
### Prerequisites
- Python 3.8 or later
- Required Python libraries (see [Installation](#installation))
- `ffmpeg` (for audio preprocessing)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/matttic97/field_recordings_search_engine.git
   cd field_recordings_search_engine
   ```

2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install `ffmpeg`:
   - On Ubuntu/Debian:
     ```bash
     sudo apt install ffmpeg
     ```
   - On macOS using Homebrew:
     ```bash
     brew install ffmpeg
     ```

4. Download and set up pre-trained models:
   - For source separation, ZFTurbo's MDX model should already be contained in configs dir.
   - For Whisper, the model will be downloaded automatically during the first run.

---

## Usage
### 1. Preprocessing
Run source separation and ASR on a folder of audio recordings:
```bash
python Preprocess/preprocess_pipeline.py --input_dir path/to/audio_dir --output_dir path/to/output_dir --text_output_dir path/to/text_output_dir
```
This separates vocals from audio files and saves them to the specified output directory. Seperated vocal audio files are then run through ASR, transcriptions are saved to the specified text output directory.

### 2. Fuzzy Search Engine
Once the audio files are processed and transcripted, they can be indexed and searched.

Run indexing on transcriptions:
```bash
python FuzzySearchEngine/index_files.py --transcriptions_dir path/to/transcriptions --index_output_dir path/to/index_output_dir --stop_words_path path/to/stop_words_text_file
```

Run fuzzy search cli:
```bash
python FuzzySearchEngine/fuzzy_search_cli.py --index_dir path/to/index_output_dir --stop_words_path path/to/stop_words_text_file
>search query:find-20:letalo
Document ID: example_787_vocals.txt, Score: 0.06521739130434782
Document ID: example_366_vocals.txt, Score: 0.006038647342995169
Document ID: example_283_vocals.txt, Score: 0.003952569169960474
Document ID: example_165_vocals.txt, Score: 0.003952569169960474
Document ID: example_936_vocals.txt, Score: 0.003450655624568668
Document ID: example_407_vocals.txt, Score: 0.003105590062111801
Document ID: example_993_vocals.txt, Score: 0.0024154589371980675
Document ID: example_95_vocals.txt, Score: 0.0024154589371980675
Document ID: example_93_vocals.txt, Score: 0.0024154589371980675
Document ID: example_873_vocals.txt, Score: 0.0024154589371980675
Document ID: example_856_vocals.txt, Score: 0.0024154589371980675
Document ID: example_751_vocals.txt, Score: 0.0024154589371980675
Document ID: example_732_vocals.txt, Score: 0.0024154589371980675
Document ID: example_68_vocals.txt, Score: 0.0024154589371980675
Document ID: example_679_vocals.txt, Score: 0.0024154589371980675
Document ID: example_603_vocals.txt, Score: 0.0024154589371980675
Document ID: example_597_vocals.txt, Score: 0.0024154589371980675
Document ID: example_550_vocals.txt, Score: 0.0024154589371980675
Document ID: example_521_vocals.txt, Score: 0.0024154589371980675
Document ID: example_494_vocals.txt, Score: 0.0024154589371980675
>search query:find-3:Slavčkov, ki smo ti zlat ven zdobil, ne?
Document ID: example_11_vocals.txt, Score: 0.22655945419103313
Document ID: example_730_vocals.txt, Score: 0.21851851851851853
Document ID: example_799_vocals.txt, Score: 0.21666666666666667
>search query:exit()
```

Fuzzy search can also be performed within Python:
```python
from FuzzySearchEngine.fuzzy_search import FuzzySearch

search_engine = FuzzySearch("path/to/index_output_dir", "path/to/stop_words_text_file")
results = search_engine.find_relevant_documents("some search query", 20)
```

---

## Acknowledgments
- [ZFTurbo](https://github.com/ZFTurbo/Music-Source-Separation-Training) for the MDX model used for source separation.
- [OpenAI](https://github.com/openai/whisper) for the Whisper ASR model.
- [benhoyt](https://github.com/benhoyt/pybktree) for the base of our BK-tree implementation.
- All contributors to open-source tools and libraries used in this project. 

---

## Contact
For questions or support, please open an issue in this repository.