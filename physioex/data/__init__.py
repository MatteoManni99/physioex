from physioex.data.base import PhysioExDataset, TimeDistributedModule
from physioex.data.dreem import Dreem
from physioex.data.mitdb import MITBIH
from physioex.data.sleep_physionet import SleepPhysionet
from physioex.data.merged import Merged

datasets = {"sleep_physionet": SleepPhysionet, "dreem": Dreem, "mitdb": MITBIH, "merged": Merged}
