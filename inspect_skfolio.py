from skfolio.optimization import MeanRisk
from skfolio.prior import BlackLitterman

mr = MeanRisk()
print("MeanRisk params:", list(mr.get_params().keys()))

bl = BlackLitterman()
print("BlackLitterman params:", list(bl.get_params().keys()))
