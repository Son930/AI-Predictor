// Redesigned PositionMonitor.jsx
import { Card } from "@heroui/react";
import { useState, useEffect } from "react";
import PositionCard from "./PositionCard";
import axios from "axios";

function PositionMonitor() {
  const [positions, setPositions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const apiUrl = "http://localhost:8000";

  // Fetch positions from API
  const fetchPositions = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await axios.get(`${apiUrl}/open_positions`);

      setPositions(response.data.positions || []);
      setLastUpdated(new Date());
    } catch (err) {
      setError(err.message || "Failed to fetch positions");
      setPositions([]);
    } finally {
      setLoading(false);
    }
  };

  // Close position handler with confirmation
  const closePosition = async (ticket) => {
    if (window.confirm("Are you sure you want to close this position?")) {
      try {
        await axios.post(`${apiUrl}/close_position/${ticket}`);
        fetchPositions(); // Refresh after closing
      } catch (err) {
        setError(
          "Failed to close position: " + (err.message || "Unknown error")
        );
      }
    }
  };

  // Fetch positions on mount and every 5 seconds
  useEffect(() => {
    fetchPositions(); // Initial fetch
    const intervalId = setInterval(fetchPositions, 5000);
    return () => clearInterval(intervalId); // Cleanup on unmount
  }, []);

  return (
    <Card className="p-6 bg-white dark:bg-gray-800 shadow-xl rounded-xl">
      {/* Header with Refresh Button */}
      <div className="flex justify-between items-center mb-4">
        <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-100">
          Open Positions
        </h1>
        <div className="flex items-center">
          {/* Show positions count */}
          <span className="mr-4 px-3 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-300 rounded-full text-sm font-medium">
            {positions.length}{" "}
            {positions.length === 1 ? "Position" : "Positions"}
          </span>
          <button
            onClick={fetchPositions}
            disabled={loading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 dark:bg-blue-500 dark:hover:bg-blue-600 transition-colors"
            aria-label="Refresh positions"
          >
            {loading ? "Refreshing..." : "Refresh Now"}
          </button>
        </div>
      </div>

      {/* Loading Spinner */}
      {loading && (
        <div className="flex justify-center items-center mb-4">
          <svg
            className="animate-spin h-5 w-5 text-blue-600 dark:text-blue-400"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            aria-label="Loading"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
            ></circle>
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0c4.418 0 8 3.582 8 8s-3.582 8-8 8V4a4 4 0 00-4 4H4z"
            ></path>
          </svg>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="mb-4 p-4 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300 rounded-md">
          {error}
        </div>
      )}

      {/* Last Updated Timestamp */}
      {lastUpdated && (
        <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
          Last updated: {lastUpdated.toLocaleTimeString()}
        </p>
      )}

      {/* Positions Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {positions.length === 0 && !loading ? (
          <div className="col-span-full text-center p-8 bg-gray-50 dark:bg-gray-700 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
            <svg
              className="mx-auto h-12 w-12 text-gray-400 dark:text-gray-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
              xmlns="http://www.w3.org/2000/svg"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M12 14a7 7 0 110-14 7 7 0 010 14z"
              ></path>
            </svg>
            <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-gray-100">
              No Open Positions
            </h3>
            <p className="mt-1 text-gray-500 dark:text-gray-400">
              There are currently no open positions to display.
            </p>
          </div>
        ) : (
          positions.map((pos) => (
            <PositionCard
              key={pos.ticket}
              position={pos}
              onClose={closePosition}
            />
          ))
        )}
      </div>
    </Card>
  );
}

export default PositionMonitor;
