from physioex.data.base import PhysioExDataset, TimeDistributedModule
from physioex.data.dreem.dreem import Dreem
from physioex.data.sleep_edf.sleep_edf import SleepEDF
from physioex.data.merged import Merged

datasets = {"sleep_physionet": SleepEDF, "dreem": Dreem, "merged": Merged}
