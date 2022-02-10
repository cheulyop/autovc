# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DementiaBank is a shared database of multimedia interactions for the study of communication in dementia. Access to the data in DementiaBank is password protected and restricted to members of the DementiaBank consortium group. This release only concerns the Pitt Corpus."""


import os
import glob
from typing import Optional

import datasets
from datasets.tasks import AutomaticSpeechRecognition


_CITATION = """\
@article{becker1994natural,
  title={The natural history of Alzheimer's disease: description of study cohort and accuracy of diagnosis},
  author={Becker, James T and Boiler, Fran{\c{c}}ois and Lopez, Oscar L and Saxton, Judith and McGonigle, Karen L},
  journal={Archives of neurology},
  volume={51},
  number={6},
  pages={585--594},
  year={1994},
  publisher={American Medical Association}
}
"""

_DESCRIPTION = """\
DementiaBank Pitt Corpus includes audios and transcripts of 99 controls and 194 dementia patients. These transcripts and audio files were gathered as part of a larger protocol administered by the Alzheimer and Related Dementias Study at the University of Pittsburgh School of Medicine. The original acquisition of the DementiaBank data was supported by NIH grants AG005133 and AG003705 to the University of Pittsburgh. Participants included elderly controls, people with probable and possible Alzheimer’s Disease, and people with other dementia diagnoses. Data were gathered longitudinally, on a yearly basis.
"""

_HOMEPAGE = "https://dementia.talkbank.org/access/English/Pitt.html"


class DementiaBank(datasets.GeneratorBasedBuilder):
    """DementiaBank Pitt Corpus"""

    VERSION = datasets.Version("0.1.0")

    def manual_download_instructions(self) -> Optional[str]:
        return (
            """\
            To use DementiaBank Pitt Corpus, data files must be downloaded manually to a local drive. You must be an authorized member of the DementiaBank consortium group to download the corpus. After downloading files to a local drive, there must be two folders with audio and transcript files: '<path>/<to>/<root>/dementia/English/Pitt/Control/cookie' and '<path>/<to>/<root>/dementia/English/Pitt/Dementia/cookie'. You must download transcript files separately as a zipped file then unzip the archive to the same folder as the audios.
            """
        )

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=datasets.Features(
                {
                    'file': datasets.Value('string'),
                    'text': datasets.Value('string'),
                    'id': datasets.Value('string'),
                    'control': datasets.Value('bool')
                }
            ),
            supervised_keys=('file', 'text'),
            homepage=_HOMEPAGE,
            citation=_CITATION,
            task_templates=[AutomaticSpeechRecognition(
                audio_file_path_column='file', transcription_column='text'
            )],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        data_dir = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))
        print(data_dir)
        return [
            datasets.SplitGenerator(
                name='full',
                gen_kwargs={
                    "ctrl_glob": os.path.join(data_dir, 'control/*.mp3'),
                    'dmnt_glob': os.path.join(data_dir, 'dementia/*.mp3'),
                }
            )
        ]

    def _generate_examples(self, ctrl_glob, dmnt_glob):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.
        ctrl_audios = glob.glob(ctrl_glob)
        dmnt_audios = glob.glob(dmnt_glob)
        for i, fname in enumerate(ctrl_audios + dmnt_audios):
                id_ = fname.split('.')[0].split(os.sep)[-1]
                example = {
                    'file': fname,
                    'text': fname.split('.')[0] + '.cha',
                    'id': id_,
                    'control': True if i < len(ctrl_audios) else False
                }
                yield id_, example