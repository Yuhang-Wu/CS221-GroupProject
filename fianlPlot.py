import oracle as oc
import EigenPortfolio as ep
import bestStock as bs
import uniformBuySell as ubs
from utils import plotAndEval as pe
import TestingData as td
import numpy as np

def main():
        TestDate, TestReturn, stockPrice, TestTimeIndex, dateSelected = td.get_testing_data()
        c = 0.0001
        D = 10
        transCostParams = {
        'c': np.array([ [c] for _ in range(D) ]),
        'c0': c
        }
        baselineTransCostParams = np.zeros(D + 1) + c
        baselineGrowthRates = 1.0 + ep.baseline(stockPrice, TestTimeIndex, baselineTransCostParams)
        oracleGrowthRates = 1.0 + np.array(oc.solveOracle(stockPrice, 10000, transCostParams['c'], transCostParams['c0']))
        bsGrowthRates = 1.0 + np.array(bs.bestStock(stockPrice))
        ubsGrowthRates = 1.0 + np.array(ubs.ubs(stockPrice))
        plotE = pe.plotEval(TestDate, 2016)
        plotE.addReturn(baselineGrowthRates, 'Baseline')
        plotE.addReturn(oracleGrowthRates, 'Oracle')
        plotE.addReturn(bsGrowthRates, 'BestStock')
        plotE.addReturn(ubsGrowthRates,'Uniform Buy & Sell')
        plotE.generatePlot()


if __name__=='__main__':
    main()