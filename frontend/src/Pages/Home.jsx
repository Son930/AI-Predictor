import { useState } from "react";
import Plot from "react-plotly.js"; // Install: npm install react-plotly.js

export default function StockDataFetcher() {
  const [ticker, setTicker] = useState("");
  const [dataType, setDataType] = useState("price");
  const [years, setYears] = useState(2); // Add years state
  const [forecastDays, setForecastDays] = useState(7); // Add forecastDays state
  const [stockData, setStockData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const fetchStockData = async () => {
    if (!ticker.trim()) return;

    setLoading(true);
    setError("");
    setStockData(null);

    try {
      const response = await fetch(
        `http://127.0.0.1:8300/stock/${ticker}?data_type=${dataType}&years=${years}&forecast_days=${forecastDays}` // Include years and forecast_days
      );

      if (!response.ok) {
        const errorData = await response.json(); // Try to get error details from the server
        throw new Error(errorData.detail || "Failed to fetch stock data"); // Use detail or default message
      }

      const data = await response.json();
      setStockData(data);
    } catch (err) {
      setError(err.message); // Display error message
      console.error("Error fetching data:", err); // Log the error for debugging
    } finally {
      setLoading(false);
    }
  };

  const handleTrainModel = async () => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(`http://127.0.0.1:8300/train_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ticker: ticker }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to train model");
      }

      const data = await response.json();
      setStockData(data);
      alert("Model trained successfully!");
    } catch (err) {
      setError(err.message);
      console.error("Error training model:", err);
    } finally {
      setLoading(false);
    }
  };

 const handlePredict = async () => {
   setLoading(true);
   setError("");

   try {
     const response = await fetch(`http://127.0.0.1:8300/predict`, {
       method: "GET",
       params: { ticker: ticker }, // Correct: Using query parameters
     }); // Removed Content-Type header

     if (!response.ok) {
       const errorData = await response.json();
       throw new Error(errorData.detail || "Failed to make prediction");
     }

     const prediction = await response.json();
     setStockData(prediction);
     alert("Prediction completed!");
   } catch (err) {
     setError(err.message);
     console.error("Error making prediction:", err);
   } finally {
     setLoading(false);
   }
 };
  const renderHistoricalData = () => {
    if (stockData && stockData.historical_data) {
      const data = stockData.historical_data;
      return (
        <div>
          <h3>Historical Data</h3>
          <table className="table-auto w-full border-collapse border border-gray-300">
            <thead>
              <tr className="bg-gray-100">
                {Object.keys(data[0]).map((key) => (
                  <th key={key} className="border border-gray-300 px-4 py-2">
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((item, index) => (
                <tr key={index} className={index % 2 === 0 ? "bg-gray-50" : ""}>
                  {Object.values(item).map((value, index) => (
                    <td
                      key={index}
                      className="border border-gray-300 px-4 py-2"
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    return null;
  };

  const renderForecastData = () => {
    if (stockData && stockData.forecast) {
      const forecastData = stockData.forecast;

      const chartData = [
        {
          x: forecastData.map((item) => item.ds),
          y: forecastData.map((item) => item.yhat),
          type: "scatter",
          mode: "lines",
          name: "Forecast",
        },
        {
          x: forecastData.map((item) => item.ds),
          y: forecastData.map((item) => item.yhat_upper),
          type: "scatter",
          mode: "lines",
          name: "Upper Bound",
          line: { dash: "dash" },
        },
        {
          x: forecastData.map((item) => item.ds),
          y: forecastData.map((item) => item.yhat_lower),
          type: "scatter",
          mode: "lines",
          name: "Lower Bound",
          line: { dash: "dash" },
        },
      ];

      return (
        <div>
          <h3>Forecast Data</h3>
          <Plot data={chartData} layout={{ title: "Stock Price Forecast" }} />
          <table className="table-auto w-full border-collapse border border-gray-300">
            <thead>
              <tr className="bg-gray-100">
                {Object.keys(forecastData[0]).map((key) => (
                  <th key={key} className="border border-gray-300 px-4 py-2">
                    {key}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {forecastData.map((item, index) => (
                <tr key={index} className={index % 2 === 0 ? "bg-gray-50" : ""}>
                  {Object.values(item).map((value, index) => (
                    <td
                      key={index}
                      className="border border-gray-300 px-4 py-2"
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center p-4 bg-gray-100">
      <h2 className="text-3xl font-bold mb-6 text-center">
        Stock Data Fetcher
      </h2>
      <div className="flex space-x-4 mb-6 w-full max-w-lg">
        <input
          type="text"
          placeholder="Enter Ticker (e.g. AAPL)"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="p-3 border rounded-md w-1/2 text-black focus:ring-2 focus:ring-blue-500"
        />
        <select
          value={dataType}
          onChange={(e) => setDataType(e.target.value)}
          className="p-3 border rounded-md w-1/2 text-black focus:ring-2 focus:ring-blue-500"
        >
          <option value="price">Real-time Price</option>
          <option value="history">Historical Data & Forecast</option>
        </select>
        {dataType === "history" && (
          <>
            <input
              type="number"
              placeholder="Years"
              value={years}
              onChange={(e) => setYears(parseInt(e.target.value) || 0)}
              className="p-3 border rounded-md w-1/4 text-black focus:ring-2 focus:ring-blue-500"
            />
            <input
              type="number"
              placeholder="Forecast Days"
              value={forecastDays}
              onChange={(e) => setForecastDays(parseInt(e.target.value) || 0)}
              className="p-3 border rounded-md w-1/4 text-black focus:ring-2 focus:ring-blue-500"
            />
          </>
        )}
        <button
          onClick={fetchStockData}
          className="p-3 bg-blue-500 text-white rounded-md hover:bg-blue-700 w-1/2 focus:ring-2 focus:ring-blue-500"
        >
          Fetch Data
        </button>
      </div>

      {/* Training and Prediction buttons */}
      <div className="flex space-x-4 mb-6 w-full max-w-lg">
        <button
          onClick={handleTrainModel}
          className="p-3 bg-green-500 text-white rounded-md hover:bg-green-700 w-1/2 focus:ring-2 focus:ring-green-500"
        >
          Train Model
        </button>
        <button
          onClick={handlePredict}
          className="p-3 bg-yellow-500 text-white rounded-md hover:bg-yellow-700 w-1/2 focus:ring-2 focus:ring-yellow-500"
        >
          Predict
        </button>
      </div>

      {loading && <p className="text-lg">Loading...</p>}
      {error && (
        <p className="text-white bg-red-700 p-2 rounded-md border">{error}</p>
      )}
      {stockData && stockData.real_time_price && (
        <div>
          <h3>Real-Time Price</h3>
          <pre className="bg-gray-800 text-green-400 p-4 rounded-md overflow-auto">
            {JSON.stringify(stockData.real_time_price, null, 2)}
          </pre>
        </div>
      )}
      {renderHistoricalData()}
      {renderForecastData()}
    </div>
  );
}
