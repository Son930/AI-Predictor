/* eslint-disable react/prop-types */
import { useState, useEffect } from "react";

export default function App() {
  const [botStatus, setBotStatus] = useState({});
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState("");

  // Fetch bot status
  const fetchStatus = async () => {
    try {
      const response = await fetch("http://localhost:8000/status");
      if (!response.ok) throw new Error("Failed to fetch status");
      const data = await response.json();
      setBotStatus(data);
      setError("");
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Control buttons
  const sendCommand = async (command) => {
    try {
      const response = await fetch(`http://localhost:8000/api/v1/${command}`, {
        method: "POST",
      });
      if (!response.ok) throw new Error("Command failed");
      await fetchStatus();
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-800 mb-8">
          AI Trading Bot Dashboard
        </h1>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Status Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Bot Status</h2>
              <div className="space-y-2">
                <StatusItem
                  label="Running"
                  value={botStatus.running ? "Yes" : "No"}
                />
                <StatusItem
                  label="Paused"
                  value={botStatus.paused ? "Yes" : "No"}
                />
                <StatusItem
                  label="Positions"
                  value={botStatus.position_count}
                />
                <StatusItem label="Errors" value={botStatus.error_count} />
                <StatusItem
                  label="Last Confidence"
                  value={
                    botStatus.last_confidence
                      ? `${(botStatus.last_confidence * 100).toFixed(1)}%`
                      : "N/A"
                  }
                />
              </div>
            </div>

            {/* Positions Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Open Positions</h2>
              {botStatus.position_count > 0 ? (
                <div className="space-y-2">
                  {/* Replace with actual position data from your API */}
                  <div className="p-2 bg-gray-50 rounded">
                    <p>EURUSD - Buy (1.0854)</p>
                    <p className="text-sm text-green-600">+1.2% Profit</p>
                  </div>
                </div>
              ) : (
                <p className="text-gray-500">No open positions</p>
              )}
            </div>

            {/* Controls Card */}
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-xl font-semibold mb-4">Controls</h2>
              <div className="grid grid-cols-2 gap-3">
                <button
                  onClick={() => sendCommand("start")}
                  className="p-2 bg-green-500 text-white rounded hover:bg-green-600 disabled:bg-gray-300"
                  disabled={botStatus.running}
                >
                  Start Bot
                </button>
                <button
                  onClick={() => sendCommand("stop")}
                  className="p-2 bg-red-500 text-white rounded hover:bg-red-600 disabled:bg-gray-300"
                  disabled={!botStatus.running}
                >
                  Stop Bot
                </button>
                <button
                  onClick={() => sendCommand("pause")}
                  className="p-2 bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:bg-gray-300"
                  disabled={!botStatus.running || botStatus.paused}
                >
                  Pause Bot
                </button>
                <button
                  onClick={() => sendCommand("resume")}
                  className="p-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
                  disabled={!botStatus.paused}
                >
                  Resume Bot
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Logs Section */}
        <div className="mt-8 bg-white rounded-lg shadow p-6">
          <h2 className="text-xl font-semibold mb-4">Logs</h2>
          <div className="h-64 overflow-y-auto bg-gray-50 p-3 rounded">
            {/* Log entries would go here */}
            <p className="text-sm text-gray-500">
              [2025-03-21 14:25:21] INFO: Bot initialized
            </p>
            <p className="text-sm text-red-500">
              [2025-03-21 14:25:22] ERROR: Connection failed
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function StatusItem({ label, value }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-gray-600">{label}:</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}
