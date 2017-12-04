import oracle as oc
import EigenPortfolio as ep
import bestStock as bs
import uniformBuySell as ubs
from utils import plotAndEval as pe


 baselineGrowthRates = 1.0 + ep.baseline(stockPrice, TestTimeIndex, baselineTransCostParams)
        oracleGrowthRates = np.array(oc.solveOracle(stockPrice, 10000, transCostParams['c'], transCostParams['c0'])) + 1.0
        bsGrowthRates = np.array(bs.bestStock(stockPrice)) + 1.0
        ubsGrowthRates = np.array(ubs.ubs(stockPrice)) + 1.0
        growthRates = growthRates[-len(baselineGrowthRates):]
        totalGR = du.prod(growthRates)
        baselineTotalGR = du.prod(baselineGrowthRates)
        plotE = pe.plotEval(TestDate, 2016)
        plotE.addReturn(baselineGrowthRates, 'Baseline')
        plotE.addReturn(growthRates, 'CNN')
        plotE.addReturn(oracleGrowthRates, 'Oracle')
        plotE.addReturn(bsGrowthRates, 'BestStock')
        plotE.addReturn(ubsGrowthRates,'Uniform Buy & Sell')
        plotE.generatePlot()
        logger.info('model total growth rate in testing data: '+ str(totalGR))
        logger.info('baseline total growth rate: '+str(baselineTotalGR))
        logger.info('')