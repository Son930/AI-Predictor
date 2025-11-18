import { useState, useEffect } from "react";

function Dashboard() {
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [serverStatus, setServerStatus] = useState("Checking...");
    const [errorMessage, setErrorMessage] = useState("");

  const port = 8500;

  useEffect(() => {
    const updateOnlineStatus = () => {
      setIsOnline(navigator.onLine);
    };

    window.addEventListener("online", updateOnlineStatus);
    window.addEventListener("offline", updateOnlineStatus);

    // Cleanup event listeners on component unmount
    return () => {
      window.removeEventListener("online", updateOnlineStatus);
      window.removeEventListener("offline", updateOnlineStatus);
    };
  }, []);

  useEffect(() => {
    const checkServerStatus = async () => {
      try {
        const response = await fetch(`http://127.0.0.1:${port}/health`);
        if (response.ok) {
          const data = await response.json();
          setServerStatus(
            data.status === "healthy" ? "Connected" : "Disconnected"
          );
        } else {
          setServerStatus("Disconnected");
        }
      } catch(err){
            setServerStatus("Disconnected");
            setErrorMessage("Error connecting to the server: " + err);
      }
    };

    checkServerStatus();
  }, [port]);

  return (
    <div className="flex justify-evenly justify-items-stretch content-around gap-6 p-4 bg-gray-800 text-white rounded-lg shadow-md mb-6">
      <p>
        <strong>Network Status:</strong> {isOnline ? "Online" : "Offline"}
      </p>
      <p>
        <strong>Server Status:</strong> {serverStatus}
      </p>
      <p>
        <strong>Port:</strong> {port}
      </p>
      {errorMessage && <div className="text-red-500">{errorMessage}</div>}
    </div>
  );
}

export default Dashboard;
