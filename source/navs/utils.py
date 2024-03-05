from source.navs import seqnav as spm
from source.navs import perfect_memory as pm
from source.navs import infomax

def pick_nav(nav_name=None):
    navs = {'smw': spm.SequentialPerfectMemory, 
            'asmw': spm.SequentialPerfectMemory, 
            's2s': spm.Seq2SeqPerfectMemory,
            'imax': infomax}
    if not navs.get(nav_name):
        raise Exception('Non valid navigator class name')
    return navs.get(nav_name)