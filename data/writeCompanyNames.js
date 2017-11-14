/*
GETTING S&P 500 TOP X COMPANIES BASED ON CAPITAL
as of Nov 13 2017
*/

var fs = require('fs')

var X = 152

function main() {
	var template = '<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="'
	var companyNames = []
	var iters = X
	var s = 0
	for (var i = 0 ; i<iters; i++) {
		let found = html.indexOf(template, s)
		companyNames.push(html.substring(found + template.length, found+template.length+6))
		s = found + template.length
	}
	//console.log(companyNames)
	var companyCodes = companyNames.map((item)=> {
		return item.substring(0, item.indexOf('"'))
	})
	console.log(companyCodes)


	var filePath = './sp500tops.txt'

	//var data2write = ['a', 'b', 'c']
	
	fs.writeFile(filePath, companyCodes.join('\n'), function(err) {})
	
	
}

var html = `<div class="panel-body">
							<table id="example-1" class="table table-striped table-bordered" cellspacing="0" width="100%">
								<thead>
									<tr>
										<th>Rank</th>
										<th>Company</th>
										<th>Symbol</th>
										<th>Weight</th>
									</tr>
								</thead>

								<tbody>
		                            
									<tr style="color: black;">
										<td>1</td>
										<td><a href="http://www.google.com/finance?q=AAPL">Apple Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AAPL"> <input type="submit" value="AAPL"> </div></form></td>
										<td>4.074237</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>2</td>
										<td><a href="http://www.google.com/finance?q=MSFT">Microsoft Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MSFT"> <input type="submit" value="MSFT"> </div></form></td>
										<td>2.917579</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>3</td>
										<td><a href="http://www.google.com/finance?q=AMZN">Amazon.com Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AMZN"> <input type="submit" value="AMZN"> </div></form></td>
										<td>2.026788</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>4</td>
										<td><a href="http://www.google.com/finance?q=FB">Facebook Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FB"> <input type="submit" value="FB"> </div></form></td>
										<td>1.910706</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>5</td>
										<td><a href="http://www.google.com/finance?q=JNJ">Johnson &amp; Johnson</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="JNJ"> <input type="submit" value="JNJ"> </div></form></td>
										<td>1.691731</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>6</td>
										<td><a href="http://www.google.com/finance?q=BRK.B">Berkshire Hathaway Inc. Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BRK.B"> <input type="submit" value="BRK.B"> </div></form></td>
										<td>1.596773</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>7</td>
										<td><a href="http://www.google.com/finance?q=XOM">Exxon Mobil Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XOM"> <input type="submit" value="XOM"> </div></form></td>
										<td>1.587610</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>8</td>
										<td><a href="http://www.google.com/finance?q=JPM">JPMorgan Chase &amp; Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="JPM"> <input type="submit" value="JPM"> </div></form></td>
										<td>1.549540</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>9</td>
										<td><a href="http://www.google.com/finance?q=GOOGL">Alphabet Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GOOGL"> <input type="submit" value="GOOGL"> </div></form></td>
										<td>1.404043</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>10</td>
										<td><a href="http://www.google.com/finance?q=GOOG">Alphabet Inc. Class C</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GOOG"> <input type="submit" value="GOOG"> </div></form></td>
										<td>1.401957</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>11</td>
										<td><a href="http://www.google.com/finance?q=BAC">Bank of America Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BAC"> <input type="submit" value="BAC"> </div></form></td>
										<td>1.174528</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>12</td>
										<td><a href="http://www.google.com/finance?q=WFC">Wells Fargo &amp; Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WFC"> <input type="submit" value="WFC"> </div></form></td>
										<td>1.085664</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>13</td>
										<td><a href="http://www.google.com/finance?q=PG">Procter &amp; Gamble Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PG"> <input type="submit" value="PG"> </div></form></td>
										<td>1.015341</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>14</td>
										<td><a href="http://www.google.com/finance?q=CVX">Chevron Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CVX"> <input type="submit" value="CVX"> </div></form></td>
										<td>1.002995</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>15</td>
										<td><a href="http://www.google.com/finance?q=INTC">Intel Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="INTC"> <input type="submit" value="INTC"> </div></form></td>
										<td>0.969746</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>16</td>
										<td><a href="http://www.google.com/finance?q=T">AT&amp;T Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="T"> <input type="submit" value="T"> </div></form></td>
										<td>0.950511</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>17</td>
										<td><a href="http://www.google.com/finance?q=PFE">Pfizer Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PFE"> <input type="submit" value="PFE"> </div></form></td>
										<td>0.948538</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>18</td>
										<td><a href="http://www.google.com/finance?q=V">Visa Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="V"> <input type="submit" value="V"> </div></form></td>
										<td>0.924260</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>19</td>
										<td><a href="http://www.google.com/finance?q=UNH">UnitedHealth Group Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UNH"> <input type="submit" value="UNH"> </div></form></td>
										<td>0.921802</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>20</td>
										<td><a href="http://www.google.com/finance?q=C">Citigroup Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="C"> <input type="submit" value="C"> </div></form></td>
										<td>0.888916</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>21</td>
										<td><a href="http://www.google.com/finance?q=HD">Home Depot Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HD"> <input type="submit" value="HD"> </div></form></td>
										<td>0.873521</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>22</td>
										<td><a href="http://www.google.com/finance?q=VZ">Verizon Communications Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VZ"> <input type="submit" value="VZ"> </div></form></td>
										<td>0.827169</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>23</td>
										<td><a href="http://www.google.com/finance?q=KO">Coca-Cola Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KO"> <input type="submit" value="KO"> </div></form></td>
										<td>0.808561</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>24</td>
										<td><a href="http://www.google.com/finance?q=GE">General Electric Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GE"> <input type="submit" value="GE"> </div></form></td>
										<td>0.801172</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>25</td>
										<td><a href="http://www.google.com/finance?q=CMCSA">Comcast Corporation Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CMCSA"> <input type="submit" value="CMCSA"> </div></form></td>
										<td>0.782825</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>26</td>
										<td><a href="http://www.google.com/finance?q=CSCO">Cisco Systems Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CSCO"> <input type="submit" value="CSCO"> </div></form></td>
										<td>0.768393</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>27</td>
										<td><a href="http://www.google.com/finance?q=DWDP">DowDuPont Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DWDP"> <input type="submit" value="DWDP"> </div></form></td>
										<td>0.737996</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>28</td>
										<td><a href="http://www.google.com/finance?q=DIS">Walt Disney Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DIS"> <input type="submit" value="DIS"> </div></form></td>
										<td>0.731328</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>29</td>
										<td><a href="http://www.google.com/finance?q=PEP">PepsiCo Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PEP"> <input type="submit" value="PEP"> </div></form></td>
										<td>0.727678</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>30</td>
										<td><a href="http://www.google.com/finance?q=PM">Philip Morris International Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PM"> <input type="submit" value="PM"> </div></form></td>
										<td>0.722900</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>31</td>
										<td><a href="http://www.google.com/finance?q=ABBV">AbbVie Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ABBV"> <input type="submit" value="ABBV"> </div></form></td>
										<td>0.686440</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>32</td>
										<td><a href="http://www.google.com/finance?q=MRK">Merck &amp; Co. Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MRK"> <input type="submit" value="MRK"> </div></form></td>
										<td>0.685394</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>33</td>
										<td><a href="http://www.google.com/finance?q=ORCL">Oracle Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ORCL"> <input type="submit" value="ORCL"> </div></form></td>
										<td>0.672771</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>34</td>
										<td><a href="http://www.google.com/finance?q=BA">Boeing Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BA"> <input type="submit" value="BA"> </div></form></td>
										<td>0.654466</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>35</td>
										<td><a href="http://www.google.com/finance?q=MA">Mastercard Incorporated Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MA"> <input type="submit" value="MA"> </div></form></td>
										<td>0.630631</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>36</td>
										<td><a href="http://www.google.com/finance?q=MMM">3M Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MMM"> <input type="submit" value="MMM"> </div></form></td>
										<td>0.614088</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>37</td>
										<td><a href="http://www.google.com/finance?q=MCD">McDonald's Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MCD"> <input type="submit" value="MCD"> </div></form></td>
										<td>0.605742</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>38</td>
										<td><a href="http://www.google.com/finance?q=WMT">Wal-Mart Stores Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WMT"> <input type="submit" value="WMT"> </div></form></td>
										<td>0.600974</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>39</td>
										<td><a href="http://www.google.com/finance?q=NVDA">NVIDIA Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NVDA"> <input type="submit" value="NVDA"> </div></form></td>
										<td>0.585858</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>40</td>
										<td><a href="http://www.google.com/finance?q=IBM">International Business Machines Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IBM"> <input type="submit" value="IBM"> </div></form></td>
										<td>0.584063</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>41</td>
										<td><a href="http://www.google.com/finance?q=AMGN">Amgen Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AMGN"> <input type="submit" value="AMGN"> </div></form></td>
										<td>0.567927</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>42</td>
										<td><a href="http://www.google.com/finance?q=MO">Altria Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MO"> <input type="submit" value="MO"> </div></form></td>
										<td>0.563358</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>43</td>
										<td><a href="http://www.google.com/finance?q=HON">Honeywell International Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HON"> <input type="submit" value="HON"> </div></form></td>
										<td>0.501989</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>44</td>
										<td><a href="http://www.google.com/finance?q=AVGO">Broadcom Limited</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AVGO"> <input type="submit" value="AVGO"> </div></form></td>
										<td>0.486706</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>45</td>
										<td><a href="http://www.google.com/finance?q=MDT">Medtronic plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MDT"> <input type="submit" value="MDT"> </div></form></td>
										<td>0.485255</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>46</td>
										<td><a href="http://www.google.com/finance?q=BMY">Bristol-Myers Squibb Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BMY"> <input type="submit" value="BMY"> </div></form></td>
										<td>0.450749</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>47</td>
										<td><a href="http://www.google.com/finance?q=GILD">Gilead Sciences Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GILD"> <input type="submit" value="GILD"> </div></form></td>
										<td>0.435219</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>48</td>
										<td><a href="http://www.google.com/finance?q=TXN">Texas Instruments Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TXN"> <input type="submit" value="TXN"> </div></form></td>
										<td>0.433429</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>49</td>
										<td><a href="http://www.google.com/finance?q=QCOM">QUALCOMM Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="QCOM"> <input type="submit" value="QCOM"> </div></form></td>
										<td>0.430526</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>50</td>
										<td><a href="http://www.google.com/finance?q=ABT">Abbott Laboratories</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ABT"> <input type="submit" value="ABT"> </div></form></td>
										<td>0.429623</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>51</td>
										<td><a href="http://www.google.com/finance?q=UNP">Union Pacific Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UNP"> <input type="submit" value="UNP"> </div></form></td>
										<td>0.421807</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>52</td>
										<td><a href="http://www.google.com/finance?q=SLB">Schlumberger NV</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SLB"> <input type="submit" value="SLB"> </div></form></td>
										<td>0.413757</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>53</td>
										<td><a href="http://www.google.com/finance?q=ADBE">Adobe Systems Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ADBE"> <input type="submit" value="ADBE"> </div></form></td>
										<td>0.405959</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>54</td>
										<td><a href="http://www.google.com/finance?q=ACN">Accenture Plc Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ACN"> <input type="submit" value="ACN"> </div></form></td>
										<td>0.401604</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>55</td>
										<td><a href="http://www.google.com/finance?q=UTX">United Technologies Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UTX"> <input type="submit" value="UTX"> </div></form></td>
										<td>0.396261</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>56</td>
										<td><a href="http://www.google.com/finance?q=GS">Goldman Sachs Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GS"> <input type="submit" value="GS"> </div></form></td>
										<td>0.390119</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>57</td>
										<td><a href="http://www.google.com/finance?q=PYPL">PayPal Holdings Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PYPL"> <input type="submit" value="PYPL"> </div></form></td>
										<td>0.377813</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>58</td>
										<td><a href="http://www.google.com/finance?q=PCLN">Priceline Group Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PCLN"> <input type="submit" value="PCLN"> </div></form></td>
										<td>0.376857</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>59</td>
										<td><a href="http://www.google.com/finance?q=SBUX">Starbucks Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SBUX"> <input type="submit" value="SBUX"> </div></form></td>
										<td>0.373203</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>60</td>
										<td><a href="http://www.google.com/finance?q=NFLX">Netflix Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NFLX"> <input type="submit" value="NFLX"> </div></form></td>
										<td>0.373081</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>61</td>
										<td><a href="http://www.google.com/finance?q=USB">U.S. Bancorp</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="USB"> <input type="submit" value="USB"> </div></form></td>
										<td>0.371213</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>62</td>
										<td><a href="http://www.google.com/finance?q=LLY">Eli Lilly and Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LLY"> <input type="submit" value="LLY"> </div></form></td>
										<td>0.365389</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>63</td>
										<td><a href="http://www.google.com/finance?q=CAT">Caterpillar Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CAT"> <input type="submit" value="CAT"> </div></form></td>
										<td>0.363277</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>64</td>
										<td><a href="http://www.google.com/finance?q=CELG">Celgene Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CELG"> <input type="submit" value="CELG"> </div></form></td>
										<td>0.361026</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>65</td>
										<td><a href="http://www.google.com/finance?q=UPS">United Parcel Service Inc. Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UPS"> <input type="submit" value="UPS"> </div></form></td>
										<td>0.351352</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>66</td>
										<td><a href="http://www.google.com/finance?q=LMT">Lockheed Martin Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LMT"> <input type="submit" value="LMT"> </div></form></td>
										<td>0.350682</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>67</td>
										<td><a href="http://www.google.com/finance?q=COST">Costco Wholesale Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="COST"> <input type="submit" value="COST"> </div></form></td>
										<td>0.339582</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>68</td>
										<td><a href="http://www.google.com/finance?q=TMO">Thermo Fisher Scientific Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TMO"> <input type="submit" value="TMO"> </div></form></td>
										<td>0.336687</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>69</td>
										<td><a href="http://www.google.com/finance?q=NKE">NIKE Inc. Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NKE"> <input type="submit" value="NKE"> </div></form></td>
										<td>0.332823</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>70</td>
										<td><a href="http://www.google.com/finance?q=NEE">NextEra Energy Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NEE"> <input type="submit" value="NEE"> </div></form></td>
										<td>0.328185</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>71</td>
										<td><a href="http://www.google.com/finance?q=CVS">CVS Health Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CVS"> <input type="submit" value="CVS"> </div></form></td>
										<td>0.326771</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>72</td>
										<td><a href="http://www.google.com/finance?q=CRM">salesforce.com inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CRM"> <input type="submit" value="CRM"> </div></form></td>
										<td>0.324682</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>73</td>
										<td><a href="http://www.google.com/finance?q=CB">Chubb Limited</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CB"> <input type="submit" value="CB"> </div></form></td>
										<td>0.318230</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>74</td>
										<td><a href="http://www.google.com/finance?q=TWX">Time Warner Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TWX"> <input type="submit" value="TWX"> </div></form></td>
										<td>0.316699</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>75</td>
										<td><a href="http://www.google.com/finance?q=MS">Morgan Stanley</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MS"> <input type="submit" value="MS"> </div></form></td>
										<td>0.311402</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>76</td>
										<td><a href="http://www.google.com/finance?q=AXP">American Express Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AXP"> <input type="submit" value="AXP"> </div></form></td>
										<td>0.309767</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>77</td>
										<td><a href="http://www.google.com/finance?q=CHTR">Charter Communications Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CHTR"> <input type="submit" value="CHTR"> </div></form></td>
										<td>0.306997</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>78</td>
										<td><a href="http://www.google.com/finance?q=BIIB">Biogen Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BIIB"> <input type="submit" value="BIIB"> </div></form></td>
										<td>0.296420</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>79</td>
										<td><a href="http://www.google.com/finance?q=LOW">Lowe's Companies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LOW"> <input type="submit" value="LOW"> </div></form></td>
										<td>0.295384</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>80</td>
										<td><a href="http://www.google.com/finance?q=CL">Colgate-Palmolive Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CL"> <input type="submit" value="CL"> </div></form></td>
										<td>0.292041</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>81</td>
										<td><a href="http://www.google.com/finance?q=COP">ConocoPhillips</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="COP"> <input type="submit" value="COP"> </div></form></td>
										<td>0.291164</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>82</td>
										<td><a href="http://www.google.com/finance?q=AMT">American Tower Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AMT"> <input type="submit" value="AMT"> </div></form></td>
										<td>0.290365</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>83</td>
										<td><a href="http://www.google.com/finance?q=PNC">PNC Financial Services Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PNC"> <input type="submit" value="PNC"> </div></form></td>
										<td>0.286680</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>84</td>
										<td><a href="http://www.google.com/finance?q=MDLZ">Mondelez International Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MDLZ"> <input type="submit" value="MDLZ"> </div></form></td>
										<td>0.285624</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>85</td>
										<td><a href="http://www.google.com/finance?q=DUK">Duke Energy Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DUK"> <input type="submit" value="DUK"> </div></form></td>
										<td>0.280805</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>86</td>
										<td><a href="http://www.google.com/finance?q=WBA">Walgreens Boots Alliance Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WBA"> <input type="submit" value="WBA"> </div></form></td>
										<td>0.278513</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>87</td>
										<td><a href="http://www.google.com/finance?q=EOG">EOG Resources Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EOG"> <input type="submit" value="EOG"> </div></form></td>
										<td>0.272756</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>88</td>
										<td><a href="http://www.google.com/finance?q=AMAT">Applied Materials Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AMAT"> <input type="submit" value="AMAT"> </div></form></td>
										<td>0.271449</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>89</td>
										<td><a href="http://www.google.com/finance?q=AET">Aetna Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AET"> <input type="submit" value="AET"> </div></form></td>
										<td>0.261876</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>90</td>
										<td><a href="http://www.google.com/finance?q=AGN">Allergan plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AGN"> <input type="submit" value="AGN"> </div></form></td>
										<td>0.261315</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>91</td>
										<td><a href="http://www.google.com/finance?q=BLK">BlackRock Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BLK"> <input type="submit" value="BLK"> </div></form></td>
										<td>0.260879</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>92</td>
										<td><a href="http://www.google.com/finance?q=ANTM">Anthem Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ANTM"> <input type="submit" value="ANTM"> </div></form></td>
										<td>0.260478</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>93</td>
										<td><a href="http://www.google.com/finance?q=DHR">Danaher Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DHR"> <input type="submit" value="DHR"> </div></form></td>
										<td>0.254037</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>94</td>
										<td><a href="http://www.google.com/finance?q=AIG">American International Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AIG"> <input type="submit" value="AIG"> </div></form></td>
										<td>0.253367</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>95</td>
										<td><a href="http://www.google.com/finance?q=GM">General Motors Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GM"> <input type="submit" value="GM"> </div></form></td>
										<td>0.252458</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>96</td>
										<td><a href="http://www.google.com/finance?q=GD">General Dynamics Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GD"> <input type="submit" value="GD"> </div></form></td>
										<td>0.251197</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>97</td>
										<td><a href="http://www.google.com/finance?q=MET">MetLife Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MET"> <input type="submit" value="MET"> </div></form></td>
										<td>0.250505</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>98</td>
										<td><a href="http://www.google.com/finance?q=RTN">Raytheon Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RTN"> <input type="submit" value="RTN"> </div></form></td>
										<td>0.244179</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>99</td>
										<td><a href="http://www.google.com/finance?q=FDX">FedEx Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FDX"> <input type="submit" value="FDX"> </div></form></td>
										<td>0.241494</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>100</td>
										<td><a href="http://www.google.com/finance?q=BK">Bank of New York Mellon Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BK"> <input type="submit" value="BK"> </div></form></td>
										<td>0.238439</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>101</td>
										<td><a href="http://www.google.com/finance?q=SCHW">Charles Schwab Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SCHW"> <input type="submit" value="SCHW"> </div></form></td>
										<td>0.238201</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>102</td>
										<td><a href="http://www.google.com/finance?q=NOC">Northrop Grumman Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NOC"> <input type="submit" value="NOC"> </div></form></td>
										<td>0.234916</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>103</td>
										<td><a href="http://www.google.com/finance?q=D">Dominion Energy Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="D"> <input type="submit" value="D"> </div></form></td>
										<td>0.234645</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>104</td>
										<td><a href="http://www.google.com/finance?q=MON">Monsanto Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MON"> <input type="submit" value="MON"> </div></form></td>
										<td>0.234539</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>105</td>
										<td><a href="http://www.google.com/finance?q=OXY">Occidental Petroleum Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="OXY"> <input type="submit" value="OXY"> </div></form></td>
										<td>0.234444</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>106</td>
										<td><a href="http://www.google.com/finance?q=SO">Southern Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SO"> <input type="submit" value="SO"> </div></form></td>
										<td>0.231879</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>107</td>
										<td><a href="http://www.google.com/finance?q=SPG">Simon Property Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SPG"> <input type="submit" value="SPG"> </div></form></td>
										<td>0.230990</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>108</td>
										<td><a href="http://www.google.com/finance?q=MU">Micron Technology Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MU"> <input type="submit" value="MU"> </div></form></td>
										<td>0.225638</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>109</td>
										<td><a href="http://www.google.com/finance?q=BDX">Becton Dickinson and Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BDX"> <input type="submit" value="BDX"> </div></form></td>
										<td>0.225193</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>110</td>
										<td><a href="http://www.google.com/finance?q=CI">Cigna Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CI"> <input type="submit" value="CI"> </div></form></td>
										<td>0.225066</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>111</td>
										<td><a href="http://www.google.com/finance?q=ADP">Automatic Data Processing Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ADP"> <input type="submit" value="ADP"> </div></form></td>
										<td>0.224955</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>112</td>
										<td><a href="http://www.google.com/finance?q=SYK">Stryker Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SYK"> <input type="submit" value="SYK"> </div></form></td>
										<td>0.224613</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>113</td>
										<td><a href="http://www.google.com/finance?q=ITW">Illinois Tool Works Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ITW"> <input type="submit" value="ITW"> </div></form></td>
										<td>0.221215</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>114</td>
										<td><a href="http://www.google.com/finance?q=ATVI">Activision Blizzard Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ATVI"> <input type="submit" value="ATVI"> </div></form></td>
										<td>0.215179</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>115</td>
										<td><a href="http://www.google.com/finance?q=KHC">Kraft Heinz Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KHC"> <input type="submit" value="KHC"> </div></form></td>
										<td>0.214899</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>116</td>
										<td><a href="http://www.google.com/finance?q=PRU">Prudential Financial Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PRU"> <input type="submit" value="PRU"> </div></form></td>
										<td>0.214277</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>117</td>
										<td><a href="http://www.google.com/finance?q=F">Ford Motor Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="F"> <input type="submit" value="F"> </div></form></td>
										<td>0.212223</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>118</td>
										<td><a href="http://www.google.com/finance?q=CME">CME Group Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CME"> <input type="submit" value="CME"> </div></form></td>
										<td>0.210570</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>119</td>
										<td><a href="http://www.google.com/finance?q=CCI">Crown Castle International Corp</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CCI"> <input type="submit" value="CCI"> </div></form></td>
										<td>0.207912</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>120</td>
										<td><a href="http://www.google.com/finance?q=CSX">CSX Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CSX"> <input type="submit" value="CSX"> </div></form></td>
										<td>0.206541</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>121</td>
										<td><a href="http://www.google.com/finance?q=TJX">TJX Companies Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TJX"> <input type="submit" value="TJX"> </div></form></td>
										<td>0.201762</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>122</td>
										<td><a href="http://www.google.com/finance?q=CTSH">Cognizant Technology Solutions Corporation Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CTSH"> <input type="submit" value="CTSH"> </div></form></td>
										<td>0.197648</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>123</td>
										<td><a href="http://www.google.com/finance?q=ISRG">Intuitive Surgical Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ISRG"> <input type="submit" value="ISRG"> </div></form></td>
										<td>0.196758</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>124</td>
										<td><a href="http://www.google.com/finance?q=MMC">Marsh &amp; McLennan Companies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MMC"> <input type="submit" value="MMC"> </div></form></td>
										<td>0.193491</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>125</td>
										<td><a href="http://www.google.com/finance?q=DE">Deere &amp; Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DE"> <input type="submit" value="DE"> </div></form></td>
										<td>0.191354</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>126</td>
										<td><a href="http://www.google.com/finance?q=PX">Praxair Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PX"> <input type="submit" value="PX"> </div></form></td>
										<td>0.190442</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>127</td>
										<td><a href="http://www.google.com/finance?q=COF">Capital One Financial Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="COF"> <input type="submit" value="COF"> </div></form></td>
										<td>0.189284</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>128</td>
										<td><a href="http://www.google.com/finance?q=KMB">Kimberly-Clark Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KMB"> <input type="submit" value="KMB"> </div></form></td>
										<td>0.183449</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>129</td>
										<td><a href="http://www.google.com/finance?q=SPGI">S&amp;P Global Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SPGI"> <input type="submit" value="SPGI"> </div></form></td>
										<td>0.183073</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>130</td>
										<td><a href="http://www.google.com/finance?q=PSX">Phillips 66</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PSX"> <input type="submit" value="PSX"> </div></form></td>
										<td>0.181690</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>131</td>
										<td><a href="http://www.google.com/finance?q=EMR">Emerson Electric Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EMR"> <input type="submit" value="EMR"> </div></form></td>
										<td>0.179415</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>132</td>
										<td><a href="http://www.google.com/finance?q=EXC">Exelon Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EXC"> <input type="submit" value="EXC"> </div></form></td>
										<td>0.179225</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>133</td>
										<td><a href="http://www.google.com/finance?q=ICE">Intercontinental Exchange Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ICE"> <input type="submit" value="ICE"> </div></form></td>
										<td>0.176122</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>134</td>
										<td><a href="http://www.google.com/finance?q=BSX">Boston Scientific Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BSX"> <input type="submit" value="BSX"> </div></form></td>
										<td>0.175458</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>135</td>
										<td><a href="http://www.google.com/finance?q=HAL">Halliburton Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HAL"> <input type="submit" value="HAL"> </div></form></td>
										<td>0.174479</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>136</td>
										<td><a href="http://www.google.com/finance?q=TRV">Travelers Companies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TRV"> <input type="submit" value="TRV"> </div></form></td>
										<td>0.171079</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>137</td>
										<td><a href="http://www.google.com/finance?q=MAR">Marriott International Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MAR"> <input type="submit" value="MAR"> </div></form></td>
										<td>0.170506</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>138</td>
										<td><a href="http://www.google.com/finance?q=STZ">Constellation Brands Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="STZ"> <input type="submit" value="STZ"> </div></form></td>
										<td>0.170426</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>139</td>
										<td><a href="http://www.google.com/finance?q=VRTX">Vertex Pharmaceuticals Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VRTX"> <input type="submit" value="VRTX"> </div></form></td>
										<td>0.169417</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>140</td>
										<td><a href="http://www.google.com/finance?q=EQIX">Equinix Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EQIX"> <input type="submit" value="EQIX"> </div></form></td>
										<td>0.169107</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>141</td>
										<td><a href="http://www.google.com/finance?q=BBT">BB&amp;T Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BBT"> <input type="submit" value="BBT"> </div></form></td>
										<td>0.169057</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>142</td>
										<td><a href="http://www.google.com/finance?q=INTU">Intuit Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="INTU"> <input type="submit" value="INTU"> </div></form></td>
										<td>0.168183</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>143</td>
										<td><a href="http://www.google.com/finance?q=NSC">Norfolk Southern Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NSC"> <input type="submit" value="NSC"> </div></form></td>
										<td>0.166727</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>144</td>
										<td><a href="http://www.google.com/finance?q=AEP">American Electric Power Company Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AEP"> <input type="submit" value="AEP"> </div></form></td>
										<td>0.165986</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>145</td>
										<td><a href="http://www.google.com/finance?q=AON">Aon plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AON"> <input type="submit" value="AON"> </div></form></td>
										<td>0.164447</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>146</td>
										<td><a href="http://www.google.com/finance?q=ALL">Allstate Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ALL"> <input type="submit" value="ALL"> </div></form></td>
										<td>0.163364</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>147</td>
										<td><a href="http://www.google.com/finance?q=VLO">Valero Energy Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VLO"> <input type="submit" value="VLO"> </div></form></td>
										<td>0.162284</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>148</td>
										<td><a href="http://www.google.com/finance?q=EBAY">eBay Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EBAY"> <input type="submit" value="EBAY"> </div></form></td>
										<td>0.161679</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>149</td>
										<td><a href="http://www.google.com/finance?q=HPQ">HP Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HPQ"> <input type="submit" value="HPQ"> </div></form></td>
										<td>0.160133</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>150</td>
										<td><a href="http://www.google.com/finance?q=HUM">Humana Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HUM"> <input type="submit" value="HUM"> </div></form></td>
										<td>0.159381</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>151</td>
										<td><a href="http://www.google.com/finance?q=ETN">Eaton Corp. Plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ETN"> <input type="submit" value="ETN"> </div></form></td>
										<td>0.158070</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>152</td>
										<td><a href="http://www.google.com/finance?q=PLD">Prologis Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PLD"> <input type="submit" value="PLD"> </div></form></td>
										<td>0.157513</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>153</td>
										<td><a href="http://www.google.com/finance?q=EA">Electronic Arts Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EA"> <input type="submit" value="EA"> </div></form></td>
										<td>0.156940</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>154</td>
										<td><a href="http://www.google.com/finance?q=APD">Air Products and Chemicals Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="APD"> <input type="submit" value="APD"> </div></form></td>
										<td>0.156787</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>155</td>
										<td><a href="http://www.google.com/finance?q=ESRX">Express Scripts Holding Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ESRX"> <input type="submit" value="ESRX"> </div></form></td>
										<td>0.156562</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>156</td>
										<td><a href="http://www.google.com/finance?q=JCI">Johnson Controls International plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="JCI"> <input type="submit" value="JCI"> </div></form></td>
										<td>0.155892</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>157</td>
										<td><a href="http://www.google.com/finance?q=ECL">Ecolab Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ECL"> <input type="submit" value="ECL"> </div></form></td>
										<td>0.154315</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>158</td>
										<td><a href="http://www.google.com/finance?q=ZTS">Zoetis Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ZTS"> <input type="submit" value="ZTS"> </div></form></td>
										<td>0.153408</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>159</td>
										<td><a href="http://www.google.com/finance?q=STT">State Street Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="STT"> <input type="submit" value="STT"> </div></form></td>
										<td>0.153279</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>160</td>
										<td><a href="http://www.google.com/finance?q=KMI">Kinder Morgan Inc Class P</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KMI"> <input type="submit" value="KMI"> </div></form></td>
										<td>0.152867</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>161</td>
										<td><a href="http://www.google.com/finance?q=LYB">LyondellBasell Industries NV</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LYB"> <input type="submit" value="LYB"> </div></form></td>
										<td>0.152787</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>162</td>
										<td><a href="http://www.google.com/finance?q=LRCX">Lam Research Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LRCX"> <input type="submit" value="LRCX"> </div></form></td>
										<td>0.151966</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>163</td>
										<td><a href="http://www.google.com/finance?q=TGT">Target Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TGT"> <input type="submit" value="TGT"> </div></form></td>
										<td>0.151441</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>164</td>
										<td><a href="http://www.google.com/finance?q=TEL">TE Connectivity Ltd.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TEL"> <input type="submit" value="TEL"> </div></form></td>
										<td>0.149265</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>165</td>
										<td><a href="http://www.google.com/finance?q=AFL">Aflac Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AFL"> <input type="submit" value="AFL"> </div></form></td>
										<td>0.148938</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>166</td>
										<td><a href="http://www.google.com/finance?q=ADI">Analog Devices Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ADI"> <input type="submit" value="ADI"> </div></form></td>
										<td>0.148544</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>167</td>
										<td><a href="http://www.google.com/finance?q=WM">Waste Management Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WM"> <input type="submit" value="WM"> </div></form></td>
										<td>0.148445</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>168</td>
										<td><a href="http://www.google.com/finance?q=DAL">Delta Air Lines Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DAL"> <input type="submit" value="DAL"> </div></form></td>
										<td>0.146776</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>169</td>
										<td><a href="http://www.google.com/finance?q=BAX">Baxter International Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BAX"> <input type="submit" value="BAX"> </div></form></td>
										<td>0.145076</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>170</td>
										<td><a href="http://www.google.com/finance?q=SHW">Sherwin-Williams Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SHW"> <input type="submit" value="SHW"> </div></form></td>
										<td>0.142749</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>171</td>
										<td><a href="http://www.google.com/finance?q=PSA">Public Storage</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PSA"> <input type="submit" value="PSA"> </div></form></td>
										<td>0.141959</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>172</td>
										<td><a href="http://www.google.com/finance?q=MPC">Marathon Petroleum Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MPC"> <input type="submit" value="MPC"> </div></form></td>
										<td>0.141057</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>173</td>
										<td><a href="http://www.google.com/finance?q=GIS">General Mills Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GIS"> <input type="submit" value="GIS"> </div></form></td>
										<td>0.139477</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>174</td>
										<td><a href="http://www.google.com/finance?q=REGN">Regeneron Pharmaceuticals Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="REGN"> <input type="submit" value="REGN"> </div></form></td>
										<td>0.138914</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>175</td>
										<td><a href="http://www.google.com/finance?q=FIS">Fidelity National Information Services Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FIS"> <input type="submit" value="FIS"> </div></form></td>
										<td>0.137678</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>176</td>
										<td><a href="http://www.google.com/finance?q=FOXA">Twenty-First Century Fox Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FOXA"> <input type="submit" value="FOXA"> </div></form></td>
										<td>0.137660</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>177</td>
										<td><a href="http://www.google.com/finance?q=ILMN">Illumina Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ILMN"> <input type="submit" value="ILMN"> </div></form></td>
										<td>0.137404</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>178</td>
										<td><a href="http://www.google.com/finance?q=SRE">Sempra Energy</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SRE"> <input type="submit" value="SRE"> </div></form></td>
										<td>0.136677</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>179</td>
										<td><a href="http://www.google.com/finance?q=PGR">Progressive Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PGR"> <input type="submit" value="PGR"> </div></form></td>
										<td>0.133195</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>180</td>
										<td><a href="http://www.google.com/finance?q=PPG">PPG Industries Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PPG"> <input type="submit" value="PPG"> </div></form></td>
										<td>0.133043</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>181</td>
										<td><a href="http://www.google.com/finance?q=LUV">Southwest Airlines Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LUV"> <input type="submit" value="LUV"> </div></form></td>
										<td>0.132774</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>182</td>
										<td><a href="http://www.google.com/finance?q=PCG">PG&amp;E Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PCG"> <input type="submit" value="PCG"> </div></form></td>
										<td>0.131377</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>183</td>
										<td><a href="http://www.google.com/finance?q=MCK">McKesson Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MCK"> <input type="submit" value="MCK"> </div></form></td>
										<td>0.131350</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>184</td>
										<td><a href="http://www.google.com/finance?q=GLW">Corning Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GLW"> <input type="submit" value="GLW"> </div></form></td>
										<td>0.129388</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>185</td>
										<td><a href="http://www.google.com/finance?q=APC">Anadarko Petroleum Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="APC"> <input type="submit" value="APC"> </div></form></td>
										<td>0.128520</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>186</td>
										<td><a href="http://www.google.com/finance?q=EL">Estee Lauder Companies Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EL"> <input type="submit" value="EL"> </div></form></td>
										<td>0.126035</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>187</td>
										<td><a href="http://www.google.com/finance?q=YUM">Yum! Brands Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="YUM"> <input type="submit" value="YUM"> </div></form></td>
										<td>0.124038</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>188</td>
										<td><a href="http://www.google.com/finance?q=STI">SunTrust Banks Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="STI"> <input type="submit" value="STI"> </div></form></td>
										<td>0.123089</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>189</td>
										<td><a href="http://www.google.com/finance?q=DXC">DXC Technology Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DXC"> <input type="submit" value="DXC"> </div></form></td>
										<td>0.122921</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>190</td>
										<td><a href="http://www.google.com/finance?q=APH">Amphenol Corporation Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="APH"> <input type="submit" value="APH"> </div></form></td>
										<td>0.122483</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>191</td>
										<td><a href="http://www.google.com/finance?q=WY">Weyerhaeuser Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WY"> <input type="submit" value="WY"> </div></form></td>
										<td>0.122193</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>192</td>
										<td><a href="http://www.google.com/finance?q=ADSK">Autodesk Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ADSK"> <input type="submit" value="ADSK"> </div></form></td>
										<td>0.122171</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>193</td>
										<td><a href="http://www.google.com/finance?q=PXD">Pioneer Natural Resources Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PXD"> <input type="submit" value="PXD"> </div></form></td>
										<td>0.121814</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>194</td>
										<td><a href="http://www.google.com/finance?q=FISV">Fiserv Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FISV"> <input type="submit" value="FISV"> </div></form></td>
										<td>0.121646</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>195</td>
										<td><a href="http://www.google.com/finance?q=CMI">Cummins Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CMI"> <input type="submit" value="CMI"> </div></form></td>
										<td>0.120284</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>196</td>
										<td><a href="http://www.google.com/finance?q=ED">Consolidated Edison Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ED"> <input type="submit" value="ED"> </div></form></td>
										<td>0.120067</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>197</td>
										<td><a href="http://www.google.com/finance?q=CCL">Carnival Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CCL"> <input type="submit" value="CCL"> </div></form></td>
										<td>0.119037</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>198</td>
										<td><a href="http://www.google.com/finance?q=SYY">Sysco Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SYY"> <input type="submit" value="SYY"> </div></form></td>
										<td>0.118830</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>199</td>
										<td><a href="http://www.google.com/finance?q=ROP">Roper Technologies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ROP"> <input type="submit" value="ROP"> </div></form></td>
										<td>0.118762</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>200</td>
										<td><a href="http://www.google.com/finance?q=EIX">Edison International</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EIX"> <input type="submit" value="EIX"> </div></form></td>
										<td>0.117748</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>201</td>
										<td><a href="http://www.google.com/finance?q=WDC">Western Digital Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WDC"> <input type="submit" value="WDC"> </div></form></td>
										<td>0.117001</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>202</td>
										<td><a href="http://www.google.com/finance?q=ROST">Ross Stores Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ROST"> <input type="submit" value="ROST"> </div></form></td>
										<td>0.116980</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>203</td>
										<td><a href="http://www.google.com/finance?q=ALXN">Alexion Pharmaceuticals Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ALXN"> <input type="submit" value="ALXN"> </div></form></td>
										<td>0.116782</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>204</td>
										<td><a href="http://www.google.com/finance?q=AVB">AvalonBay Communities Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AVB"> <input type="submit" value="AVB"> </div></form></td>
										<td>0.116009</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>205</td>
										<td><a href="http://www.google.com/finance?q=DLPH">Delphi Automotive PLC</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DLPH"> <input type="submit" value="DLPH"> </div></form></td>
										<td>0.115866</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>206</td>
										<td><a href="http://www.google.com/finance?q=XEL">Xcel Energy Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XEL"> <input type="submit" value="XEL"> </div></form></td>
										<td>0.115156</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>207</td>
										<td><a href="http://www.google.com/finance?q=PEG">Public Service Enterprise Group Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PEG"> <input type="submit" value="PEG"> </div></form></td>
										<td>0.115128</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>208</td>
										<td><a href="http://www.google.com/finance?q=EQR">Equity Residential</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EQR"> <input type="submit" value="EQR"> </div></form></td>
										<td>0.114762</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>209</td>
										<td><a href="http://www.google.com/finance?q=MNST">Monster Beverage Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MNST"> <input type="submit" value="MNST"> </div></form></td>
										<td>0.114637</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>210</td>
										<td><a href="http://www.google.com/finance?q=HCN">Welltower Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HCN"> <input type="submit" value="HCN"> </div></form></td>
										<td>0.112691</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>211</td>
										<td><a href="http://www.google.com/finance?q=ROK">Rockwell Automation Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ROK"> <input type="submit" value="ROK"> </div></form></td>
										<td>0.112421</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>212</td>
										<td><a href="http://www.google.com/finance?q=SWK">Stanley Black &amp; Decker Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SWK"> <input type="submit" value="SWK"> </div></form></td>
										<td>0.112265</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>213</td>
										<td><a href="http://www.google.com/finance?q=DLR">Digital Realty Trust Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DLR"> <input type="submit" value="DLR"> </div></form></td>
										<td>0.112194</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>214</td>
										<td><a href="http://www.google.com/finance?q=PPL">PPL Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PPL"> <input type="submit" value="PPL"> </div></form></td>
										<td>0.111344</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>215</td>
										<td><a href="http://www.google.com/finance?q=PCAR">PACCAR Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PCAR"> <input type="submit" value="PCAR"> </div></form></td>
										<td>0.110229</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>216</td>
										<td><a href="http://www.google.com/finance?q=MTB">M&amp;T Bank Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MTB"> <input type="submit" value="MTB"> </div></form></td>
										<td>0.109301</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>217</td>
										<td><a href="http://www.google.com/finance?q=PH">Parker-Hannifin Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PH"> <input type="submit" value="PH"> </div></form></td>
										<td>0.109211</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>218</td>
										<td><a href="http://www.google.com/finance?q=DFS">Discover Financial Services</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DFS"> <input type="submit" value="DFS"> </div></form></td>
										<td>0.108870</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>219</td>
										<td><a href="http://www.google.com/finance?q=BCR">C. R. Bard Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BCR"> <input type="submit" value="BCR"> </div></form></td>
										<td>0.108854</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>220</td>
										<td><a href="http://www.google.com/finance?q=SYF">Synchrony Financial</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SYF"> <input type="submit" value="SYF"> </div></form></td>
										<td>0.108226</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>221</td>
										<td><a href="http://www.google.com/finance?q=MCO">Moody's Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MCO"> <input type="submit" value="MCO"> </div></form></td>
										<td>0.107130</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>222</td>
										<td><a href="http://www.google.com/finance?q=WMB">Williams Companies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WMB"> <input type="submit" value="WMB"> </div></form></td>
										<td>0.106710</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>223</td>
										<td><a href="http://www.google.com/finance?q=AMP">Ameriprise Financial Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AMP"> <input type="submit" value="AMP"> </div></form></td>
										<td>0.106431</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>224</td>
										<td><a href="http://www.google.com/finance?q=VTR">Ventas Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VTR"> <input type="submit" value="VTR"> </div></form></td>
										<td>0.103424</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>225</td>
										<td><a href="http://www.google.com/finance?q=ADM">Archer-Daniels-Midland Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ADM"> <input type="submit" value="ADM"> </div></form></td>
										<td>0.102924</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>226</td>
										<td><a href="http://www.google.com/finance?q=HPE">Hewlett Packard Enterprise Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HPE"> <input type="submit" value="HPE"> </div></form></td>
										<td>0.102717</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>227</td>
										<td><a href="http://www.google.com/finance?q=HCA">HCA Healthcare Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HCA"> <input type="submit" value="HCA"> </div></form></td>
										<td>0.102489</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>228</td>
										<td><a href="http://www.google.com/finance?q=IP">International Paper Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IP"> <input type="submit" value="IP"> </div></form></td>
										<td>0.102322</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>229</td>
										<td><a href="http://www.google.com/finance?q=DLTR">Dollar Tree Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DLTR"> <input type="submit" value="DLTR"> </div></form></td>
										<td>0.101648</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>230</td>
										<td><a href="http://www.google.com/finance?q=TROW">T. Rowe Price Group</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TROW"> <input type="submit" value="TROW"> </div></form></td>
										<td>0.101581</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>231</td>
										<td><a href="http://www.google.com/finance?q=RHT">Red Hat Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RHT"> <input type="submit" value="RHT"> </div></form></td>
										<td>0.100955</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>232</td>
										<td><a href="http://www.google.com/finance?q=VFC">V.F. Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VFC"> <input type="submit" value="VFC"> </div></form></td>
										<td>0.100808</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>233</td>
										<td><a href="http://www.google.com/finance?q=ZBH">Zimmer Biomet Holdings Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ZBH"> <input type="submit" value="ZBH"> </div></form></td>
										<td>0.100545</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>234</td>
										<td><a href="http://www.google.com/finance?q=DG">Dollar General Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DG"> <input type="submit" value="DG"> </div></form></td>
										<td>0.099078</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>235</td>
										<td><a href="http://www.google.com/finance?q=EW">Edwards Lifesciences Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EW"> <input type="submit" value="EW"> </div></form></td>
										<td>0.098912</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>236</td>
										<td><a href="http://www.google.com/finance?q=FTV">Fortive Corp.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FTV"> <input type="submit" value="FTV"> </div></form></td>
										<td>0.098725</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>237</td>
										<td><a href="http://www.google.com/finance?q=WLTW">Willis Towers Watson Public Limited Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WLTW"> <input type="submit" value="WLTW"> </div></form></td>
										<td>0.098105</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>238</td>
										<td><a href="http://www.google.com/finance?q=COL">Rockwell Collins Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="COL"> <input type="submit" value="COL"> </div></form></td>
										<td>0.097882</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>239</td>
										<td><a href="http://www.google.com/finance?q=A">Agilent Technologies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="A"> <input type="submit" value="A"> </div></form></td>
										<td>0.097356</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>240</td>
										<td><a href="http://www.google.com/finance?q=IR">Ingersoll-Rand Plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IR"> <input type="submit" value="IR"> </div></form></td>
										<td>0.097323</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>241</td>
										<td><a href="http://www.google.com/finance?q=TSN">Tyson Foods Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TSN"> <input type="submit" value="TSN"> </div></form></td>
										<td>0.096972</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>242</td>
										<td><a href="http://www.google.com/finance?q=WEC">WEC Energy Group Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WEC"> <input type="submit" value="WEC"> </div></form></td>
										<td>0.096835</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>243</td>
										<td><a href="http://www.google.com/finance?q=RCL">Royal Caribbean Cruises Ltd.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RCL"> <input type="submit" value="RCL"> </div></form></td>
										<td>0.096577</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>244</td>
										<td><a href="http://www.google.com/finance?q=DVN">Devon Energy Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DVN"> <input type="submit" value="DVN"> </div></form></td>
										<td>0.095976</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>245</td>
										<td><a href="http://www.google.com/finance?q=CXO">Concho Resources Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CXO"> <input type="submit" value="CXO"> </div></form></td>
										<td>0.095806</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>246</td>
										<td><a href="http://www.google.com/finance?q=MCHP">Microchip Technology Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MCHP"> <input type="submit" value="MCHP"> </div></form></td>
										<td>0.094897</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>247</td>
										<td><a href="http://www.google.com/finance?q=FITB">Fifth Third Bancorp</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FITB"> <input type="submit" value="FITB"> </div></form></td>
										<td>0.093664</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>248</td>
										<td><a href="http://www.google.com/finance?q=PAYX">Paychex Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PAYX"> <input type="submit" value="PAYX"> </div></form></td>
										<td>0.093566</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>249</td>
										<td><a href="http://www.google.com/finance?q=CBS">CBS Corporation Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CBS"> <input type="submit" value="CBS"> </div></form></td>
										<td>0.093281</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>250</td>
										<td><a href="http://www.google.com/finance?q=CERN">Cerner Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CERN"> <input type="submit" value="CERN"> </div></form></td>
										<td>0.093051</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>251</td>
										<td><a href="http://www.google.com/finance?q=MYL">Mylan N.V.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MYL"> <input type="submit" value="MYL"> </div></form></td>
										<td>0.092701</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>252</td>
										<td><a href="http://www.google.com/finance?q=SWKS">Skyworks Solutions Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SWKS"> <input type="submit" value="SWKS"> </div></form></td>
										<td>0.092162</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>253</td>
										<td><a href="http://www.google.com/finance?q=HIG">Hartford Financial Services Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HIG"> <input type="submit" value="HIG"> </div></form></td>
										<td>0.092155</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>254</td>
										<td><a href="http://www.google.com/finance?q=SBAC">SBA Communications Corp. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SBAC"> <input type="submit" value="SBAC"> </div></form></td>
										<td>0.091816</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>255</td>
										<td><a href="http://www.google.com/finance?q=DTE">DTE Energy Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DTE"> <input type="submit" value="DTE"> </div></form></td>
										<td>0.091095</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>256</td>
										<td><a href="http://www.google.com/finance?q=ES">Eversource Energy</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ES"> <input type="submit" value="ES"> </div></form></td>
										<td>0.090784</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>257</td>
										<td><a href="http://www.google.com/finance?q=OKE">ONEOK Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="OKE"> <input type="submit" value="OKE"> </div></form></td>
										<td>0.089967</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>258</td>
										<td><a href="http://www.google.com/finance?q=NTRS">Northern Trust Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NTRS"> <input type="submit" value="NTRS"> </div></form></td>
										<td>0.089816</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>259</td>
										<td><a href="http://www.google.com/finance?q=KR">Kroger Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KR"> <input type="submit" value="KR"> </div></form></td>
										<td>0.089460</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>260</td>
										<td><a href="http://www.google.com/finance?q=AAL">American Airlines Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AAL"> <input type="submit" value="AAL"> </div></form></td>
										<td>0.089447</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>261</td>
										<td><a href="http://www.google.com/finance?q=BXP">Boston Properties Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BXP"> <input type="submit" value="BXP"> </div></form></td>
										<td>0.087475</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>262</td>
										<td><a href="http://www.google.com/finance?q=KEY">KeyCorp</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KEY"> <input type="submit" value="KEY"> </div></form></td>
										<td>0.087344</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>263</td>
										<td><a href="http://www.google.com/finance?q=FCX">Freeport-McMoRan Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FCX"> <input type="submit" value="FCX"> </div></form></td>
										<td>0.085868</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>264</td>
										<td><a href="http://www.google.com/finance?q=NEM">Newmont Mining Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NEM"> <input type="submit" value="NEM"> </div></form></td>
										<td>0.085575</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>265</td>
										<td><a href="http://www.google.com/finance?q=ORLY">O'Reilly Automotive Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ORLY"> <input type="submit" value="ORLY"> </div></form></td>
										<td>0.085180</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>266</td>
										<td><a href="http://www.google.com/finance?q=CFG">Citizens Financial Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CFG"> <input type="submit" value="CFG"> </div></form></td>
										<td>0.084281</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>267</td>
										<td><a href="http://www.google.com/finance?q=CAH">Cardinal Health Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CAH"> <input type="submit" value="CAH"> </div></form></td>
										<td>0.083358</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>268</td>
										<td><a href="http://www.google.com/finance?q=RF">Regions Financial Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RF"> <input type="submit" value="RF"> </div></form></td>
										<td>0.082516</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>269</td>
										<td><a href="http://www.google.com/finance?q=PFG">Principal Financial Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PFG"> <input type="submit" value="PFG"> </div></form></td>
										<td>0.081761</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>270</td>
										<td><a href="http://www.google.com/finance?q=XLNX">Xilinx Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XLNX"> <input type="submit" value="XLNX"> </div></form></td>
										<td>0.081550</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>271</td>
										<td><a href="http://www.google.com/finance?q=ALGN">Align Technology Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ALGN"> <input type="submit" value="ALGN"> </div></form></td>
										<td>0.081193</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>272</td>
										<td><a href="http://www.google.com/finance?q=NUE">Nucor Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NUE"> <input type="submit" value="NUE"> </div></form></td>
										<td>0.080356</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>273</td>
										<td><a href="http://www.google.com/finance?q=INCY">Incyte Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="INCY"> <input type="submit" value="INCY"> </div></form></td>
										<td>0.079929</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>274</td>
										<td><a href="http://www.google.com/finance?q=SYMC">Symantec Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SYMC"> <input type="submit" value="SYMC"> </div></form></td>
										<td>0.079612</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>275</td>
										<td><a href="http://www.google.com/finance?q=AZO">AutoZone Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AZO"> <input type="submit" value="AZO"> </div></form></td>
										<td>0.077741</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>276</td>
										<td><a href="http://www.google.com/finance?q=MGM">MGM Resorts International</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MGM"> <input type="submit" value="MGM"> </div></form></td>
										<td>0.076985</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>277</td>
										<td><a href="http://www.google.com/finance?q=CLX">Clorox Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CLX"> <input type="submit" value="CLX"> </div></form></td>
										<td>0.076046</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>278</td>
										<td><a href="http://www.google.com/finance?q=ESS">Essex Property Trust Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ESS"> <input type="submit" value="ESS"> </div></form></td>
										<td>0.075992</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>279</td>
										<td><a href="http://www.google.com/finance?q=LNC">Lincoln National Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LNC"> <input type="submit" value="LNC"> </div></form></td>
										<td>0.075643</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>280</td>
										<td><a href="http://www.google.com/finance?q=APA">Apache Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="APA"> <input type="submit" value="APA"> </div></form></td>
										<td>0.075359</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>281</td>
										<td><a href="http://www.google.com/finance?q=HRS">Harris Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HRS"> <input type="submit" value="HRS"> </div></form></td>
										<td>0.075032</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>282</td>
										<td><a href="http://www.google.com/finance?q=MHK">Mohawk Industries Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MHK"> <input type="submit" value="MHK"> </div></form></td>
										<td>0.074563</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>283</td>
										<td><a href="http://www.google.com/finance?q=VMC">Vulcan Materials Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VMC"> <input type="submit" value="VMC"> </div></form></td>
										<td>0.073607</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>284</td>
										<td><a href="http://www.google.com/finance?q=MTD">Mettler-Toledo International Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MTD"> <input type="submit" value="MTD"> </div></form></td>
										<td>0.073261</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>285</td>
										<td><a href="http://www.google.com/finance?q=CNC">Centene Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CNC"> <input type="submit" value="CNC"> </div></form></td>
										<td>0.072920</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>286</td>
										<td><a href="http://www.google.com/finance?q=K">Kellogg Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="K"> <input type="submit" value="K"> </div></form></td>
										<td>0.072829</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>287</td>
										<td><a href="http://www.google.com/finance?q=KLAC">KLA-Tencor Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KLAC"> <input type="submit" value="KLAC"> </div></form></td>
										<td>0.072478</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>288</td>
										<td><a href="http://www.google.com/finance?q=OMC">Omnicom Group Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="OMC"> <input type="submit" value="OMC"> </div></form></td>
										<td>0.071867</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>289</td>
										<td><a href="http://www.google.com/finance?q=DPS">Dr Pepper Snapple Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DPS"> <input type="submit" value="DPS"> </div></form></td>
										<td>0.071793</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>290</td>
										<td><a href="http://www.google.com/finance?q=AWK">American Water Works Company Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AWK"> <input type="submit" value="AWK"> </div></form></td>
										<td>0.071438</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>291</td>
										<td><a href="http://www.google.com/finance?q=Q">Quintiles IMS Holdings Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="Q"> <input type="submit" value="Q"> </div></form></td>
										<td>0.071366</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>292</td>
										<td><a href="http://www.google.com/finance?q=AME">AMETEK Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AME"> <input type="submit" value="AME"> </div></form></td>
										<td>0.071313</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>293</td>
										<td><a href="http://www.google.com/finance?q=INFO">IHS Markit Ltd.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="INFO"> <input type="submit" value="INFO"> </div></form></td>
										<td>0.070619</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>294</td>
										<td><a href="http://www.google.com/finance?q=EQT">EQT Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EQT"> <input type="submit" value="EQT"> </div></form></td>
										<td>0.070558</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>295</td>
										<td><a href="http://www.google.com/finance?q=DHI">D.R. Horton Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DHI"> <input type="submit" value="DHI"> </div></form></td>
										<td>0.070252</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>296</td>
										<td><a href="http://www.google.com/finance?q=ALB">Albemarle Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ALB"> <input type="submit" value="ALB"> </div></form></td>
										<td>0.070213</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>297</td>
										<td><a href="http://www.google.com/finance?q=ANDV">Andeavor</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ANDV"> <input type="submit" value="ANDV"> </div></form></td>
										<td>0.069823</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>298</td>
										<td><a href="http://www.google.com/finance?q=WAT">Waters Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WAT"> <input type="submit" value="WAT"> </div></form></td>
										<td>0.069693</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>299</td>
										<td><a href="http://www.google.com/finance?q=BBY">Best Buy Co. Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BBY"> <input type="submit" value="BBY"> </div></form></td>
										<td>0.069307</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>300</td>
										<td><a href="http://www.google.com/finance?q=ETR">Entergy Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ETR"> <input type="submit" value="ETR"> </div></form></td>
										<td>0.068946</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>301</td>
										<td><a href="http://www.google.com/finance?q=HSY">Hershey Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HSY"> <input type="submit" value="HSY"> </div></form></td>
										<td>0.068939</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>302</td>
										<td><a href="http://www.google.com/finance?q=O">Realty Income Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="O"> <input type="submit" value="O"> </div></form></td>
										<td>0.068905</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>303</td>
										<td><a href="http://www.google.com/finance?q=LH">Laboratory Corporation of America Holdings</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LH"> <input type="submit" value="LH"> </div></form></td>
										<td>0.068807</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>304</td>
										<td><a href="http://www.google.com/finance?q=XRAY">DENTSPLY SIRONA Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XRAY"> <input type="submit" value="XRAY"> </div></form></td>
										<td>0.068627</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>305</td>
										<td><a href="http://www.google.com/finance?q=UAL">United Continental Holdings Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UAL"> <input type="submit" value="UAL"> </div></form></td>
										<td>0.068223</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>306</td>
										<td><a href="http://www.google.com/finance?q=GPN">Global Payments Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GPN"> <input type="submit" value="GPN"> </div></form></td>
										<td>0.067836</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>307</td>
										<td><a href="http://www.google.com/finance?q=FE">FirstEnergy Corp.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FE"> <input type="submit" value="FE"> </div></form></td>
										<td>0.067711</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>308</td>
										<td><a href="http://www.google.com/finance?q=AEE">Ameren Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AEE"> <input type="submit" value="AEE"> </div></form></td>
										<td>0.067704</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>309</td>
										<td><a href="http://www.google.com/finance?q=CTL">CenturyLink Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CTL"> <input type="submit" value="CTL"> </div></form></td>
										<td>0.067547</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>310</td>
										<td><a href="http://www.google.com/finance?q=TAP">Molson Coors Brewing Company Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TAP"> <input type="submit" value="TAP"> </div></form></td>
										<td>0.067023</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>311</td>
										<td><a href="http://www.google.com/finance?q=HLT">Hilton Worldwide Holdings Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HLT"> <input type="submit" value="HLT"> </div></form></td>
										<td>0.066916</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>312</td>
										<td><a href="http://www.google.com/finance?q=HST">Host Hotels &amp; Resorts Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HST"> <input type="submit" value="HST"> </div></form></td>
										<td>0.066807</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>313</td>
										<td><a href="http://www.google.com/finance?q=MSI">Motorola Solutions Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MSI"> <input type="submit" value="MSI"> </div></form></td>
										<td>0.066761</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>314</td>
										<td><a href="http://www.google.com/finance?q=WRK">WestRock Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WRK"> <input type="submit" value="WRK"> </div></form></td>
										<td>0.066146</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>315</td>
										<td><a href="http://www.google.com/finance?q=RSG">Republic Services Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RSG"> <input type="submit" value="RSG"> </div></form></td>
										<td>0.066000</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>316</td>
										<td><a href="http://www.google.com/finance?q=DOV">Dover Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DOV"> <input type="submit" value="DOV"> </div></form></td>
										<td>0.065956</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>317</td>
										<td><a href="http://www.google.com/finance?q=EXPE">Expedia Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EXPE"> <input type="submit" value="EXPE"> </div></form></td>
										<td>0.065614</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>318</td>
										<td><a href="http://www.google.com/finance?q=TXT">Textron Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TXT"> <input type="submit" value="TXT"> </div></form></td>
										<td>0.065346</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>319</td>
										<td><a href="http://www.google.com/finance?q=IVZ">Invesco Ltd.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IVZ"> <input type="submit" value="IVZ"> </div></form></td>
										<td>0.065048</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>320</td>
										<td><a href="http://www.google.com/finance?q=CAG">Conagra Brands Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CAG"> <input type="submit" value="CAG"> </div></form></td>
										<td>0.064979</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>321</td>
										<td><a href="http://www.google.com/finance?q=LLL">L3 Technologies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LLL"> <input type="submit" value="LLL"> </div></form></td>
										<td>0.064827</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>322</td>
										<td><a href="http://www.google.com/finance?q=NWL">Newell Brands Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NWL"> <input type="submit" value="NWL"> </div></form></td>
										<td>0.063955</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>323</td>
										<td><a href="http://www.google.com/finance?q=HBAN">Huntington Bancshares Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HBAN"> <input type="submit" value="HBAN"> </div></form></td>
										<td>0.063712</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>324</td>
										<td><a href="http://www.google.com/finance?q=BHGE">Baker Hughes a GE Company Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BHGE"> <input type="submit" value="BHGE"> </div></form></td>
										<td>0.063341</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>325</td>
										<td><a href="http://www.google.com/finance?q=BLL">Ball Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BLL"> <input type="submit" value="BLL"> </div></form></td>
										<td>0.063240</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>326</td>
										<td><a href="http://www.google.com/finance?q=GGP">GGP Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GGP"> <input type="submit" value="GGP"> </div></form></td>
										<td>0.062882</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>327</td>
										<td><a href="http://www.google.com/finance?q=COG">Cabot Oil &amp; Gas Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="COG"> <input type="submit" value="COG"> </div></form></td>
										<td>0.062220</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>328</td>
										<td><a href="http://www.google.com/finance?q=VRSK">Verisk Analytics Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VRSK"> <input type="submit" value="VRSK"> </div></form></td>
										<td>0.062207</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>329</td>
										<td><a href="http://www.google.com/finance?q=NBL">Noble Energy Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NBL"> <input type="submit" value="NBL"> </div></form></td>
										<td>0.061855</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>330</td>
										<td><a href="http://www.google.com/finance?q=L">Loews Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="L"> <input type="submit" value="L"> </div></form></td>
										<td>0.061569</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>331</td>
										<td><a href="http://www.google.com/finance?q=FAST">Fastenal Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FAST"> <input type="submit" value="FAST"> </div></form></td>
										<td>0.061417</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>332</td>
										<td><a href="http://www.google.com/finance?q=CMS">CMS Energy Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CMS"> <input type="submit" value="CMS"> </div></form></td>
										<td>0.061415</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>333</td>
										<td><a href="http://www.google.com/finance?q=KMX">CarMax Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KMX"> <input type="submit" value="KMX"> </div></form></td>
										<td>0.061380</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>334</td>
										<td><a href="http://www.google.com/finance?q=BEN">Franklin Resources Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BEN"> <input type="submit" value="BEN"> </div></form></td>
										<td>0.061078</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>335</td>
										<td><a href="http://www.google.com/finance?q=MLM">Martin Marietta Materials Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MLM"> <input type="submit" value="MLM"> </div></form></td>
										<td>0.061022</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>336</td>
										<td><a href="http://www.google.com/finance?q=EMN">Eastman Chemical Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EMN"> <input type="submit" value="EMN"> </div></form></td>
										<td>0.060255</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>337</td>
										<td><a href="http://www.google.com/finance?q=TDG">TransDigm Group Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TDG"> <input type="submit" value="TDG"> </div></form></td>
										<td>0.059996</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>338</td>
										<td><a href="http://www.google.com/finance?q=CMA">Comerica Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CMA"> <input type="submit" value="CMA"> </div></form></td>
										<td>0.059493</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>339</td>
										<td><a href="http://www.google.com/finance?q=MRO">Marathon Oil Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MRO"> <input type="submit" value="MRO"> </div></form></td>
										<td>0.059128</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>340</td>
										<td><a href="http://www.google.com/finance?q=SNPS">Synopsys Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SNPS"> <input type="submit" value="SNPS"> </div></form></td>
										<td>0.059020</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>341</td>
										<td><a href="http://www.google.com/finance?q=EFX">Equifax Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EFX"> <input type="submit" value="EFX"> </div></form></td>
										<td>0.058852</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>342</td>
										<td><a href="http://www.google.com/finance?q=VNO">Vornado Realty Trust</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VNO"> <input type="submit" value="VNO"> </div></form></td>
										<td>0.058742</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>343</td>
										<td><a href="http://www.google.com/finance?q=IDXX">IDEXX Laboratories Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IDXX"> <input type="submit" value="IDXX"> </div></form></td>
										<td>0.058623</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>344</td>
										<td><a href="http://www.google.com/finance?q=CBOE">Cboe Global Markets Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CBOE"> <input type="submit" value="CBOE"> </div></form></td>
										<td>0.058531</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>345</td>
										<td><a href="http://www.google.com/finance?q=ANSS">ANSYS Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ANSS"> <input type="submit" value="ANSS"> </div></form></td>
										<td>0.057713</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>346</td>
										<td><a href="http://www.google.com/finance?q=NOV">National Oilwell Varco Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NOV"> <input type="submit" value="NOV"> </div></form></td>
										<td>0.057307</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>347</td>
										<td><a href="http://www.google.com/finance?q=FTI">TechnipFMC Plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FTI"> <input type="submit" value="FTI"> </div></form></td>
										<td>0.057293</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>348</td>
										<td><a href="http://www.google.com/finance?q=GPC">Genuine Parts Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GPC"> <input type="submit" value="GPC"> </div></form></td>
										<td>0.057089</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>349</td>
										<td><a href="http://www.google.com/finance?q=DGX">Quest Diagnostics Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DGX"> <input type="submit" value="DGX"> </div></form></td>
										<td>0.057087</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>350</td>
										<td><a href="http://www.google.com/finance?q=CDNS">Cadence Design Systems Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CDNS"> <input type="submit" value="CDNS"> </div></form></td>
										<td>0.056884</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>351</td>
										<td><a href="http://www.google.com/finance?q=NTAP">NetApp Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NTAP"> <input type="submit" value="NTAP"> </div></form></td>
										<td>0.056662</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>352</td>
										<td><a href="http://www.google.com/finance?q=HES">Hess Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HES"> <input type="submit" value="HES"> </div></form></td>
										<td>0.056531</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>353</td>
										<td><a href="http://www.google.com/finance?q=CTAS">Cintas Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CTAS"> <input type="submit" value="CTAS"> </div></form></td>
										<td>0.056407</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>354</td>
										<td><a href="http://www.google.com/finance?q=HCP">HCP Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HCP"> <input type="submit" value="HCP"> </div></form></td>
										<td>0.056344</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>355</td>
										<td><a href="http://www.google.com/finance?q=CNP">CenterPoint Energy Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CNP"> <input type="submit" value="CNP"> </div></form></td>
										<td>0.056278</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>356</td>
										<td><a href="http://www.google.com/finance?q=ABC">AmerisourceBergen Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ABC"> <input type="submit" value="ABC"> </div></form></td>
										<td>0.056257</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>357</td>
										<td><a href="http://www.google.com/finance?q=CBG">CBRE Group Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CBG"> <input type="submit" value="CBG"> </div></form></td>
										<td>0.056093</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>358</td>
										<td><a href="http://www.google.com/finance?q=MAS">Masco Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MAS"> <input type="submit" value="MAS"> </div></form></td>
										<td>0.055818</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>359</td>
										<td><a href="http://www.google.com/finance?q=WYNN">Wynn Resorts Limited</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WYNN"> <input type="submit" value="WYNN"> </div></form></td>
										<td>0.055713</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>360</td>
										<td><a href="http://www.google.com/finance?q=FOX">Twenty-First Century Fox Inc. Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FOX"> <input type="submit" value="FOX"> </div></form></td>
										<td>0.055559</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>361</td>
										<td><a href="http://www.google.com/finance?q=URI">United Rentals Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="URI"> <input type="submit" value="URI"> </div></form></td>
										<td>0.055471</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>362</td>
										<td><a href="http://www.google.com/finance?q=FMC">FMC Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FMC"> <input type="submit" value="FMC"> </div></form></td>
										<td>0.055448</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>363</td>
										<td><a href="http://www.google.com/finance?q=NLSN">Nielsen Holdings Plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NLSN"> <input type="submit" value="NLSN"> </div></form></td>
										<td>0.055147</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>364</td>
										<td><a href="http://www.google.com/finance?q=CTXS">Citrix Systems Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CTXS"> <input type="submit" value="CTXS"> </div></form></td>
										<td>0.054697</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>365</td>
										<td><a href="http://www.google.com/finance?q=TSS">Total System Services Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TSS"> <input type="submit" value="TSS"> </div></form></td>
										<td>0.054510</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>366</td>
										<td><a href="http://www.google.com/finance?q=WHR">Whirlpool Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WHR"> <input type="submit" value="WHR"> </div></form></td>
										<td>0.054448</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>367</td>
										<td><a href="http://www.google.com/finance?q=UNM">Unum Group</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UNM"> <input type="submit" value="UNM"> </div></form></td>
										<td>0.054232</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>368</td>
										<td><a href="http://www.google.com/finance?q=LEN">Lennar Corporation Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LEN"> <input type="submit" value="LEN"> </div></form></td>
										<td>0.053881</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>369</td>
										<td><a href="http://www.google.com/finance?q=LB">L Brands Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LB"> <input type="submit" value="LB"> </div></form></td>
										<td>0.053814</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>370</td>
										<td><a href="http://www.google.com/finance?q=XYL">Xylem Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XYL"> <input type="submit" value="XYL"> </div></form></td>
										<td>0.053782</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>371</td>
										<td><a href="http://www.google.com/finance?q=ETFC">E*TRADE Financial Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ETFC"> <input type="submit" value="ETFC"> </div></form></td>
										<td>0.053747</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>372</td>
										<td><a href="http://www.google.com/finance?q=MAA">Mid-America Apartment Communities Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MAA"> <input type="submit" value="MAA"> </div></form></td>
										<td>0.053409</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>373</td>
										<td><a href="http://www.google.com/finance?q=SJM">J. M. Smucker Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SJM"> <input type="submit" value="SJM"> </div></form></td>
										<td>0.053385</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>374</td>
										<td><a href="http://www.google.com/finance?q=RMD">ResMed Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RMD"> <input type="submit" value="RMD"> </div></form></td>
										<td>0.052853</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>375</td>
										<td><a href="http://www.google.com/finance?q=XEC">Cimarex Energy Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XEC"> <input type="submit" value="XEC"> </div></form></td>
										<td>0.052760</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>376</td>
										<td><a href="http://www.google.com/finance?q=ULTA">Ulta Beauty Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ULTA"> <input type="submit" value="ULTA"> </div></form></td>
										<td>0.052751</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>377</td>
										<td><a href="http://www.google.com/finance?q=PRGO">Perrigo Co. Plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PRGO"> <input type="submit" value="PRGO"> </div></form></td>
										<td>0.052454</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>378</td>
										<td><a href="http://www.google.com/finance?q=MKC">McCormick &amp; Company Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MKC"> <input type="submit" value="MKC"> </div></form></td>
										<td>0.052106</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>379</td>
										<td><a href="http://www.google.com/finance?q=AJG">Arthur J. Gallagher &amp; Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AJG"> <input type="submit" value="AJG"> </div></form></td>
										<td>0.051963</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>380</td>
										<td><a href="http://www.google.com/finance?q=IFF">International Flavors &amp; Fragrances Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IFF"> <input type="submit" value="IFF"> </div></form></td>
										<td>0.051920</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>381</td>
										<td><a href="http://www.google.com/finance?q=TPR">Tapestry Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TPR"> <input type="submit" value="TPR"> </div></form></td>
										<td>0.051844</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>382</td>
										<td><a href="http://www.google.com/finance?q=DISH">DISH Network Corporation Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DISH"> <input type="submit" value="DISH"> </div></form></td>
										<td>0.051821</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>383</td>
										<td><a href="http://www.google.com/finance?q=CHD">Church &amp; Dwight Co. Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CHD"> <input type="submit" value="CHD"> </div></form></td>
										<td>0.051701</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>384</td>
										<td><a href="http://www.google.com/finance?q=ARE">Alexandria Real Estate Equities Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ARE"> <input type="submit" value="ARE"> </div></form></td>
										<td>0.051587</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>385</td>
										<td><a href="http://www.google.com/finance?q=PNR">Pentair plc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PNR"> <input type="submit" value="PNR"> </div></form></td>
										<td>0.051193</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>386</td>
										<td><a href="http://www.google.com/finance?q=CHRW">C.H. Robinson Worldwide Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CHRW"> <input type="submit" value="CHRW"> </div></form></td>
										<td>0.051134</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>387</td>
										<td><a href="http://www.google.com/finance?q=LKQ">LKQ Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LKQ"> <input type="submit" value="LKQ"> </div></form></td>
										<td>0.050851</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>388</td>
										<td><a href="http://www.google.com/finance?q=BF.B">Brown-Forman Corporation Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BF.B"> <input type="submit" value="BF.B"> </div></form></td>
										<td>0.050764</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>389</td>
										<td><a href="http://www.google.com/finance?q=KSU">Kansas City Southern</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KSU"> <input type="submit" value="KSU"> </div></form></td>
										<td>0.050576</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>390</td>
										<td><a href="http://www.google.com/finance?q=WYN">Wyndham Worldwide Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WYN"> <input type="submit" value="WYN"> </div></form></td>
										<td>0.050251</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>391</td>
										<td><a href="http://www.google.com/finance?q=STX">Seagate Technology PLC</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="STX"> <input type="submit" value="STX"> </div></form></td>
										<td>0.050046</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>392</td>
										<td><a href="http://www.google.com/finance?q=COO">Cooper Companies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="COO"> <input type="submit" value="COO"> </div></form></td>
										<td>0.049952</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>393</td>
										<td><a href="http://www.google.com/finance?q=GWW">W.W. Grainger Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GWW"> <input type="submit" value="GWW"> </div></form></td>
										<td>0.049733</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>394</td>
										<td><a href="http://www.google.com/finance?q=CINF">Cincinnati Financial Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CINF"> <input type="submit" value="CINF"> </div></form></td>
										<td>0.049526</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>395</td>
										<td><a href="http://www.google.com/finance?q=HOLX">Hologic Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HOLX"> <input type="submit" value="HOLX"> </div></form></td>
										<td>0.049299</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>396</td>
										<td><a href="http://www.google.com/finance?q=ADS">Alliance Data Systems Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ADS"> <input type="submit" value="ADS"> </div></form></td>
										<td>0.049016</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>397</td>
										<td><a href="http://www.google.com/finance?q=RJF">Raymond James Financial Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RJF"> <input type="submit" value="RJF"> </div></form></td>
										<td>0.048998</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>398</td>
										<td><a href="http://www.google.com/finance?q=IT">Gartner Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IT"> <input type="submit" value="IT"> </div></form></td>
										<td>0.048718</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>399</td>
										<td><a href="http://www.google.com/finance?q=IRM">Iron Mountain Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IRM"> <input type="submit" value="IRM"> </div></form></td>
										<td>0.048625</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>400</td>
										<td><a href="http://www.google.com/finance?q=EXPD">Expeditors International of Washington Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EXPD"> <input type="submit" value="EXPD"> </div></form></td>
										<td>0.048490</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>401</td>
										<td><a href="http://www.google.com/finance?q=XL">XL Group Ltd</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XL"> <input type="submit" value="XL"> </div></form></td>
										<td>0.048121</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>402</td>
										<td><a href="http://www.google.com/finance?q=HSIC">Henry Schein Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HSIC"> <input type="submit" value="HSIC"> </div></form></td>
										<td>0.047931</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>403</td>
										<td><a href="http://www.google.com/finance?q=EXR">Extra Space Storage Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EXR"> <input type="submit" value="EXR"> </div></form></td>
										<td>0.047885</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>404</td>
										<td><a href="http://www.google.com/finance?q=AMG">Affiliated Managers Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AMG"> <input type="submit" value="AMG"> </div></form></td>
										<td>0.047208</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>405</td>
										<td><a href="http://www.google.com/finance?q=UDR">UDR Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UDR"> <input type="submit" value="UDR"> </div></form></td>
										<td>0.047208</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>406</td>
										<td><a href="http://www.google.com/finance?q=BWA">BorgWarner Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BWA"> <input type="submit" value="BWA"> </div></form></td>
										<td>0.046924</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>407</td>
										<td><a href="http://www.google.com/finance?q=PKG">Packaging Corporation of America</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PKG"> <input type="submit" value="PKG"> </div></form></td>
										<td>0.046611</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>408</td>
										<td><a href="http://www.google.com/finance?q=DRI">Darden Restaurants Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DRI"> <input type="submit" value="DRI"> </div></form></td>
										<td>0.046429</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>409</td>
										<td><a href="http://www.google.com/finance?q=PVH">PVH Corp.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PVH"> <input type="submit" value="PVH"> </div></form></td>
										<td>0.046427</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>410</td>
										<td><a href="http://www.google.com/finance?q=ARNC">Arconic Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ARNC"> <input type="submit" value="ARNC"> </div></form></td>
										<td>0.046266</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>411</td>
										<td><a href="http://www.google.com/finance?q=DRE">Duke Realty Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DRE"> <input type="submit" value="DRE"> </div></form></td>
										<td>0.046212</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>412</td>
										<td><a href="http://www.google.com/finance?q=SLG">SL Green Realty Corp.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SLG"> <input type="submit" value="SLG"> </div></form></td>
										<td>0.045952</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>413</td>
										<td><a href="http://www.google.com/finance?q=CA">CA Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CA"> <input type="submit" value="CA"> </div></form></td>
										<td>0.045923</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>414</td>
										<td><a href="http://www.google.com/finance?q=HAS">Hasbro Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HAS"> <input type="submit" value="HAS"> </div></form></td>
										<td>0.045654</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>415</td>
										<td><a href="http://www.google.com/finance?q=VAR">Varian Medical Systems Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VAR"> <input type="submit" value="VAR"> </div></form></td>
										<td>0.045568</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>416</td>
										<td><a href="http://www.google.com/finance?q=QRVO">Qorvo Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="QRVO"> <input type="submit" value="QRVO"> </div></form></td>
										<td>0.045488</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>417</td>
										<td><a href="http://www.google.com/finance?q=LNT">Alliant Energy Corp</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LNT"> <input type="submit" value="LNT"> </div></form></td>
										<td>0.044958</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>418</td>
										<td><a href="http://www.google.com/finance?q=VRSN">VeriSign Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VRSN"> <input type="submit" value="VRSN"> </div></form></td>
										<td>0.044091</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>419</td>
										<td><a href="http://www.google.com/finance?q=REG">Regency Centers Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="REG"> <input type="submit" value="REG"> </div></form></td>
										<td>0.043801</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>420</td>
										<td><a href="http://www.google.com/finance?q=NCLH">Norwegian Cruise Line Holdings Ltd.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NCLH"> <input type="submit" value="NCLH"> </div></form></td>
										<td>0.043779</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>421</td>
										<td><a href="http://www.google.com/finance?q=PNW">Pinnacle West Capital Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PNW"> <input type="submit" value="PNW"> </div></form></td>
										<td>0.043632</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>422</td>
										<td><a href="http://www.google.com/finance?q=FBHS">Fortune Brands Home &amp; Security Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FBHS"> <input type="submit" value="FBHS"> </div></form></td>
										<td>0.043548</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>423</td>
										<td><a href="http://www.google.com/finance?q=FRT">Federal Realty Investment Trust</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FRT"> <input type="submit" value="FRT"> </div></form></td>
										<td>0.042962</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>424</td>
										<td><a href="http://www.google.com/finance?q=JNPR">Juniper Networks Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="JNPR"> <input type="submit" value="JNPR"> </div></form></td>
										<td>0.042695</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>425</td>
										<td><a href="http://www.google.com/finance?q=RE">Everest Re Group Ltd.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RE"> <input type="submit" value="RE"> </div></form></td>
										<td>0.042643</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>426</td>
										<td><a href="http://www.google.com/finance?q=AKAM">Akamai Technologies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AKAM"> <input type="submit" value="AKAM"> </div></form></td>
										<td>0.042459</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>427</td>
										<td><a href="http://www.google.com/finance?q=WU">Western Union Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="WU"> <input type="submit" value="WU"> </div></form></td>
										<td>0.042195</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>428</td>
										<td><a href="http://www.google.com/finance?q=TIF">Tiffany &amp; Co.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TIF"> <input type="submit" value="TIF"> </div></form></td>
										<td>0.042190</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>429</td>
										<td><a href="http://www.google.com/finance?q=TMK">Torchmark Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TMK"> <input type="submit" value="TMK"> </div></form></td>
										<td>0.042170</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>430</td>
										<td><a href="http://www.google.com/finance?q=AVY">Avery Dennison Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AVY"> <input type="submit" value="AVY"> </div></form></td>
										<td>0.041840</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>431</td>
										<td><a href="http://www.google.com/finance?q=CPB">Campbell Soup Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CPB"> <input type="submit" value="CPB"> </div></form></td>
										<td>0.041359</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>432</td>
										<td><a href="http://www.google.com/finance?q=AMD">Advanced Micro Devices Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AMD"> <input type="submit" value="AMD"> </div></form></td>
										<td>0.040960</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>433</td>
										<td><a href="http://www.google.com/finance?q=JBHT">J.B. Hunt Transport Services Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="JBHT"> <input type="submit" value="JBHT"> </div></form></td>
										<td>0.040316</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>434</td>
										<td><a href="http://www.google.com/finance?q=SNA">Snap-on Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SNA"> <input type="submit" value="SNA"> </div></form></td>
										<td>0.040304</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>435</td>
										<td><a href="http://www.google.com/finance?q=PHM">PulteGroup Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PHM"> <input type="submit" value="PHM"> </div></form></td>
										<td>0.040205</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>436</td>
										<td><a href="http://www.google.com/finance?q=ZION">Zions Bancorporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ZION"> <input type="submit" value="ZION"> </div></form></td>
										<td>0.040001</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>437</td>
										<td><a href="http://www.google.com/finance?q=NI">NiSource Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NI"> <input type="submit" value="NI"> </div></form></td>
										<td>0.039751</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>438</td>
										<td><a href="http://www.google.com/finance?q=VIAB">Viacom Inc. Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="VIAB"> <input type="submit" value="VIAB"> </div></form></td>
										<td>0.039615</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>439</td>
										<td><a href="http://www.google.com/finance?q=NRG">NRG Energy Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NRG"> <input type="submit" value="NRG"> </div></form></td>
										<td>0.039450</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>440</td>
										<td><a href="http://www.google.com/finance?q=UHS">Universal Health Services Inc. Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UHS"> <input type="submit" value="UHS"> </div></form></td>
										<td>0.039111</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>441</td>
										<td><a href="http://www.google.com/finance?q=SEE">Sealed Air Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SEE"> <input type="submit" value="SEE"> </div></form></td>
										<td>0.039062</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>442</td>
										<td><a href="http://www.google.com/finance?q=CF">CF Industries Holdings Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CF"> <input type="submit" value="CF"> </div></form></td>
										<td>0.038814</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>443</td>
										<td><a href="http://www.google.com/finance?q=HRL">Hormel Foods Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HRL"> <input type="submit" value="HRL"> </div></form></td>
										<td>0.038707</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>444</td>
										<td><a href="http://www.google.com/finance?q=AOS">A. O. Smith Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AOS"> <input type="submit" value="AOS"> </div></form></td>
										<td>0.038696</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>445</td>
										<td><a href="http://www.google.com/finance?q=DVA">DaVita Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DVA"> <input type="submit" value="DVA"> </div></form></td>
										<td>0.038639</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>446</td>
										<td><a href="http://www.google.com/finance?q=LUK">Leucadia National Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LUK"> <input type="submit" value="LUK"> </div></form></td>
										<td>0.038067</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>447</td>
										<td><a href="http://www.google.com/finance?q=HOG">Harley-Davidson Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HOG"> <input type="submit" value="HOG"> </div></form></td>
										<td>0.037649</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>448</td>
										<td><a href="http://www.google.com/finance?q=KORS">Michael Kors Holdings Ltd</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KORS"> <input type="submit" value="KORS"> </div></form></td>
										<td>0.037441</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>449</td>
										<td><a href="http://www.google.com/finance?q=NDAQ">Nasdaq Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NDAQ"> <input type="submit" value="NDAQ"> </div></form></td>
										<td>0.037431</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>450</td>
										<td><a href="http://www.google.com/finance?q=TSCO">Tractor Supply Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TSCO"> <input type="submit" value="TSCO"> </div></form></td>
										<td>0.037279</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>451</td>
										<td><a href="http://www.google.com/finance?q=KIM">Kimco Realty Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KIM"> <input type="submit" value="KIM"> </div></form></td>
										<td>0.035882</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>452</td>
										<td><a href="http://www.google.com/finance?q=MOS">Mosaic Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MOS"> <input type="submit" value="MOS"> </div></form></td>
										<td>0.035803</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>453</td>
										<td><a href="http://www.google.com/finance?q=ALLE">Allegion PLC</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ALLE"> <input type="submit" value="ALLE"> </div></form></td>
										<td>0.035544</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>454</td>
										<td><a href="http://www.google.com/finance?q=FFIV">F5 Networks Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FFIV"> <input type="submit" value="FFIV"> </div></form></td>
										<td>0.034987</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>455</td>
										<td><a href="http://www.google.com/finance?q=COTY">Coty Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="COTY"> <input type="submit" value="COTY"> </div></form></td>
										<td>0.034636</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>456</td>
										<td><a href="http://www.google.com/finance?q=PKI">PerkinElmer Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PKI"> <input type="submit" value="PKI"> </div></form></td>
										<td>0.034466</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>457</td>
										<td><a href="http://www.google.com/finance?q=ALK">Alaska Air Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="ALK"> <input type="submit" value="ALK"> </div></form></td>
										<td>0.034330</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>458</td>
										<td><a href="http://www.google.com/finance?q=KSS">Kohl's Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="KSS"> <input type="submit" value="KSS"> </div></form></td>
										<td>0.034213</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>459</td>
										<td><a href="http://www.google.com/finance?q=IPG">Interpublic Group of Companies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="IPG"> <input type="submit" value="IPG"> </div></form></td>
										<td>0.033575</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>460</td>
										<td><a href="http://www.google.com/finance?q=SNI">Scripps Networks Interactive Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SNI"> <input type="submit" value="SNI"> </div></form></td>
										<td>0.033445</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>461</td>
										<td><a href="http://www.google.com/finance?q=GT">Goodyear Tire &amp; Rubber Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GT"> <input type="submit" value="GT"> </div></form></td>
										<td>0.033017</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>462</td>
										<td><a href="http://www.google.com/finance?q=JEC">Jacobs Engineering Group Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="JEC"> <input type="submit" value="JEC"> </div></form></td>
										<td>0.032712</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>463</td>
										<td><a href="http://www.google.com/finance?q=HBI">Hanesbrands Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HBI"> <input type="submit" value="HBI"> </div></form></td>
										<td>0.032569</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>464</td>
										<td><a href="http://www.google.com/finance?q=GRMN">Garmin Ltd.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GRMN"> <input type="submit" value="GRMN"> </div></form></td>
										<td>0.031744</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>465</td>
										<td><a href="http://www.google.com/finance?q=MAC">Macerich Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MAC"> <input type="submit" value="MAC"> </div></form></td>
										<td>0.031600</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>466</td>
										<td><a href="http://www.google.com/finance?q=CMG">Chipotle Mexican Grill Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CMG"> <input type="submit" value="CMG"> </div></form></td>
										<td>0.031561</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>467</td>
										<td><a href="http://www.google.com/finance?q=AIV">Apartment Investment and Management Company Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AIV"> <input type="submit" value="AIV"> </div></form></td>
										<td>0.031442</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>468</td>
										<td><a href="http://www.google.com/finance?q=AYI">Acuity Brands Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AYI"> <input type="submit" value="AYI"> </div></form></td>
										<td>0.031063</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>469</td>
										<td><a href="http://www.google.com/finance?q=RHI">Robert Half International Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RHI"> <input type="submit" value="RHI"> </div></form></td>
										<td>0.030805</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>470</td>
										<td><a href="http://www.google.com/finance?q=AES">AES Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AES"> <input type="submit" value="AES"> </div></form></td>
										<td>0.029996</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>471</td>
										<td><a href="http://www.google.com/finance?q=NFX">Newfield Exploration Company</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NFX"> <input type="submit" value="NFX"> </div></form></td>
										<td>0.029365</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>472</td>
										<td><a href="http://www.google.com/finance?q=FLR">Fluor Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FLR"> <input type="submit" value="FLR"> </div></form></td>
										<td>0.029020</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>473</td>
										<td><a href="http://www.google.com/finance?q=FLIR">FLIR Systems Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FLIR"> <input type="submit" value="FLIR"> </div></form></td>
										<td>0.028543</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>474</td>
										<td><a href="http://www.google.com/finance?q=XRX">Xerox Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="XRX"> <input type="submit" value="XRX"> </div></form></td>
										<td>0.027948</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>475</td>
										<td><a href="http://www.google.com/finance?q=HP">Helmerich &amp; Payne Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HP"> <input type="submit" value="HP"> </div></form></td>
										<td>0.027935</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>476</td>
										<td><a href="http://www.google.com/finance?q=SCG">SCANA Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SCG"> <input type="submit" value="SCG"> </div></form></td>
										<td>0.027927</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>477</td>
										<td><a href="http://www.google.com/finance?q=LEG">Leggett &amp; Platt Incorporated</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LEG"> <input type="submit" value="LEG"> </div></form></td>
										<td>0.027703</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>478</td>
										<td><a href="http://www.google.com/finance?q=PBCT">People's United Financial Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PBCT"> <input type="submit" value="PBCT"> </div></form></td>
										<td>0.027551</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>479</td>
										<td><a href="http://www.google.com/finance?q=M">Macy's Inc</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="M"> <input type="submit" value="M"> </div></form></td>
										<td>0.027539</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>480</td>
										<td><a href="http://www.google.com/finance?q=GPS">Gap Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="GPS"> <input type="submit" value="GPS"> </div></form></td>
										<td>0.027310</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>481</td>
										<td><a href="http://www.google.com/finance?q=AAP">Advance Auto Parts Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AAP"> <input type="submit" value="AAP"> </div></form></td>
										<td>0.026529</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>482</td>
										<td><a href="http://www.google.com/finance?q=NWSA">News Corporation Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NWSA"> <input type="submit" value="NWSA"> </div></form></td>
										<td>0.025541</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>483</td>
										<td><a href="http://www.google.com/finance?q=BHF">Brighthouse Financial Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="BHF"> <input type="submit" value="BHF"> </div></form></td>
										<td>0.025371</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>484</td>
										<td><a href="http://www.google.com/finance?q=AIZ">Assurant Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="AIZ"> <input type="submit" value="AIZ"> </div></form></td>
										<td>0.024651</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>485</td>
										<td><a href="http://www.google.com/finance?q=PWR">Quanta Services Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PWR"> <input type="submit" value="PWR"> </div></form></td>
										<td>0.024201</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>486</td>
										<td><a href="http://www.google.com/finance?q=SRCL">Stericycle Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SRCL"> <input type="submit" value="SRCL"> </div></form></td>
										<td>0.024190</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>487</td>
										<td><a href="http://www.google.com/finance?q=HRB">H&amp;R Block Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="HRB"> <input type="submit" value="HRB"> </div></form></td>
										<td>0.023443</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>488</td>
										<td><a href="http://www.google.com/finance?q=FLS">Flowserve Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FLS"> <input type="submit" value="FLS"> </div></form></td>
										<td>0.022648</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>489</td>
										<td><a href="http://www.google.com/finance?q=RL">Ralph Lauren Corporation Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RL"> <input type="submit" value="RL"> </div></form></td>
										<td>0.022340</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>490</td>
										<td><a href="http://www.google.com/finance?q=MAT">Mattel Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="MAT"> <input type="submit" value="MAT"> </div></form></td>
										<td>0.022109</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>491</td>
										<td><a href="http://www.google.com/finance?q=CSRA">CSRA Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CSRA"> <input type="submit" value="CSRA"> </div></form></td>
										<td>0.021326</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>492</td>
										<td><a href="http://www.google.com/finance?q=JWN">Nordstrom Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="JWN"> <input type="submit" value="JWN"> </div></form></td>
										<td>0.020949</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>493</td>
										<td><a href="http://www.google.com/finance?q=RRC">Range Resources Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="RRC"> <input type="submit" value="RRC"> </div></form></td>
										<td>0.019678</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>494</td>
										<td><a href="http://www.google.com/finance?q=SIG">Signet Jewelers Limited</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="SIG"> <input type="submit" value="SIG"> </div></form></td>
										<td>0.019283</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>495</td>
										<td><a href="http://www.google.com/finance?q=FL">Foot Locker Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="FL"> <input type="submit" value="FL"> </div></form></td>
										<td>0.018476</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>496</td>
										<td><a href="http://www.google.com/finance?q=CHK">Chesapeake Energy Corporation</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="CHK"> <input type="submit" value="CHK"> </div></form></td>
										<td>0.017036</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>497</td>
										<td><a href="http://www.google.com/finance?q=NAVI">Navient Corp</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NAVI"> <input type="submit" value="NAVI"> </div></form></td>
										<td>0.016694</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>498</td>
										<td><a href="http://www.google.com/finance?q=TRIP">TripAdvisor Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="TRIP"> <input type="submit" value="TRIP"> </div></form></td>
										<td>0.015820</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>499</td>
										<td><a href="http://www.google.com/finance?q=DISCK">Discovery Communications Inc. Class C</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DISCK"> <input type="submit" value="DISCK"> </div></form></td>
										<td>0.015698</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>500</td>
										<td><a href="http://www.google.com/finance?q=EVHC">Envision Healthcare Corp.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="EVHC"> <input type="submit" value="EVHC"> </div></form></td>
										<td>0.013059</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>501</td>
										<td><a href="http://www.google.com/finance?q=PDCO">Patterson Companies Inc.</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="PDCO"> <input type="submit" value="PDCO"> </div></form></td>
										<td>0.012410</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>502</td>
										<td><a href="http://www.google.com/finance?q=DISCA">Discovery Communications Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="DISCA"> <input type="submit" value="DISCA"> </div></form></td>
										<td>0.011481</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>503</td>
										<td><a href="http://www.google.com/finance?q=UAA">Under Armour Inc. Class A</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UAA"> <input type="submit" value="UAA"> </div></form></td>
										<td>0.010120</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>504</td>
										<td><a href="http://www.google.com/finance?q=UA">Under Armour Inc. Class C</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="UA"> <input type="submit" value="UA"> </div></form></td>
										<td>0.009252</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>505</td>
										<td><a href="http://www.google.com/finance?q=NWS">News Corporation Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="NWS"> <input type="submit" value="NWS"> </div></form></td>
										<td>0.007643</td>
									</tr>
		                            
									<tr style="color: black;">
										<td>506</td>
										<td><a href="http://www.google.com/finance?q=LEN.B">Lennar Corporation Class B</a></td>
										<td><form action="/charts" method="post"> <div><input type="hidden" name="symbol" value="LEN.B"> <input type="submit" value="LEN.B"> </div></form></td>
										<td>0.000908</td>
									</tr>
		                            
								</tbody>
							</table>

							<p style="color: green; font-style: italic;"> Data as of 11/10/2017</p>

							<p style="color: blue; font-style: italic;">
								Note 1: There are 506 symbols due to several companies with two share classes.
								For example, Google's parent company Alphabet has Class A (GOOGL) and Class C (GOOG) shares in the index.
							</p>

							<p style="color: blue; font-style: italic;">
								Note 2: When companies are removed and added to the index the membership list may
								temporarily show both the removed company and added company.
							</p>

						</div>`
main()
