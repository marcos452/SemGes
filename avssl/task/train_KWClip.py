from ..model import KWClip_GeneralTransformer
from .base_task import TrainSpeechClipBaseTask


class TrainKWClip_GeneralTransformer(TrainSpeechClipBaseTask):
    def __init__(self):
        super().__init__()

    def run(self):
        super().run(KWClip_GeneralTransformer)
