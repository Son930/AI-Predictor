// Redesigned RealtimeStock.jsx
/* eslint-disable react/prop-types */
import { useState, useEffect } from "react";
import Plot from "react-plotly.js";

function RealTimeStockGraph({ ticker }) {
  const [stockData, setStockData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchStockData = async () => {
      setLoading(true);
      setError("");
      try {
        const response = await fetch(
          `https://api.tiingo.com/tiingo/daily/${ticker}/prices?token=861e7708bad94378c6b680a8fc481253bab5db5a`
        );
        const data = await response.json();
        if (data["Time Series (1min)"]) {
          const timeSeries = data["Time Series (1min)"];
          const parsedData = Object.keys(timeSeries).map((time) => ({
            time,
            price: parseFloat(timeSeries[time]["1. open"]),
          }));
          setStockData(parsedData.reverse());
        } else {
          setError(data["Error Message"] || "Failed to fetch stock data");
        }
      } catch (err) {
        setError("An error occurred while fetching stock data" + err);
      } finally {
        setLoading(false);
      }
    };

    fetchStockData();
    const interval = setInterval(fetchStockData, 60000); // Fetch new data every minute

    return () => clearInterval(interval);
  }, [ticker]);

  if (loading)
    return <p className="text-gray-600 dark:text-gray-400">Loading...</p>;
  if (error) return <p className="text-red-500 dark:text-red-400">{error}</p>;

  return (
    <Plot
      data={[
        {
          x: stockData.map((data) => data.time),
          y: stockData.map((data) => data.price),
          type: "scatter",
          mode: "lines",
          line: { color: "#3b82f6" },
        },
      ]}
      layout={{
        title: `${ticker} Real-Time Stock Price`,
        xaxis: { title: "Time" },
        yaxis: { title: "Price (USD)" },
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: { color: "#374151 dark:#e5e7eb" },
      }}
      style={{ width: "100%", height: "100%" }}
    />
  );
}

export default RealTimeStockGraph;
