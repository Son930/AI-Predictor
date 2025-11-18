import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import {
  Card,
  CardBody,
  CardHeader,
  Input,
  Button,
  Alert,
  Spinner,
} from "@heroui/react";
import { ArrowLeftIcon, ArrowRightCircleIcon } from "@heroicons/react/24/solid";
import { motion } from "framer-motion";

function Mt5Login() {
  const [login, setLogin] = useState("");
  const [password, setPassword] = useState("");
  const [server, setServer] = useState("");
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const apiUrl = "http://localhost:8000";

  const handleSubmit = async () => {
    setLoading(true);
    setMessage("");
    try {
      const response = await fetch(`${apiUrl}/connect_mt5`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ login: parseInt(login), password, server }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Connection failed");
      }
      const data = await response.json();
      setMessage(data.message);
      setTimeout(() => navigate("/"), 2000);
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto min-h-screen relative p-4">
      {message && (
        <Alert
          description={message}
          title={message.startsWith("Error:") ? "Error" : "Success"}
          radius="full"
          className={`mb-4 p-4 ${
            message.startsWith("Error:")
              ? "bg-red-200 text-red-700"
              : "bg-green-200 text-green-700"
          }`}
        />
      )}
      {loading && (
        <div className="absolute inset-0 flex justify-center items-center bg-gray-200 bg-opacity-50">
          <Spinner size="lg" />
        </div>
      )}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="p-6 bg-gray-200 shadow-2xl shadow-indigo-500/50 max-w-md mx-auto">
          <CardHeader className="flex justify-between items-center">
            <h2 className="text-2xl font-bold text-gray-900 tracking-tight">
              MT5 Login
            </h2>
            <Link
              to="/"
              className="text-blue-600 hover:underline flex items-center space-x-1"
            >
              <ArrowLeftIcon className="w-5 h-5" />
              <span>Back to Dashboard</span>
            </Link>
          </CardHeader>
          <CardBody>
            <div className="space-y-4">
              <div>
                <label
                  htmlFor="login-input"
                  className="block text-sm font-medium text-gray-800 mb-2"
                >
                  Login ID
                </label>
                <Input
                  id="login-input"
                  type="number"
                  value={login}
                  onChange={(e) => setLogin(e.target.value)}
                  className="w-full"
                  variant="bordered"
                  color="default"
                  placeholder="Enter MT5 Login ID"
                  isDisabled={loading}
                  aria-label="MT5 Login ID"
                />
              </div>
              <div>
                <label
                  htmlFor="password-input"
                  className="block text-sm font-medium text-gray-800 mb-2"
                >
                  Password
                </label>
                <Input
                  id="password-input"
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full"
                  variant="bordered"
                  color="default"
                  placeholder="Enter MT5 Password"
                  isDisabled={loading}
                  aria-label="MT5 Password"
                />
              </div>
              <div>
                <label
                  htmlFor="server-input"
                  className="block text-sm font-medium text-gray-800 mb-2"
                >
                  Server
                </label>
                <Input
                  id="server-input"
                  type="text"
                  value={server}
                  onChange={(e) => setServer(e.target.value)}
                  className="w-full"
                  variant="bordered"
                  color="default"
                  placeholder="Enter MT5 Server (e.g., MetaQuotes-Demo)"
                  isDisabled={loading}
                  aria-label="MT5 Server"
                />
              </div>
              <Button
                onPress={handleSubmit}
                isDisabled={loading || !login || !password || !server}
                color="primary"
                radius="full"
                variant="solid"
                className="w-full flex items-center justify-center space-x-2"
                aria-label="Connect to MT5"
              >
                <ArrowRightCircleIcon className="w-5 h-5" />
                <span>{loading ? "Connecting..." : "Connect to MT5"}</span>
              </Button>
            </div>
          </CardBody>
        </Card>
      </motion.div>
    </div>
  );
}

export default Mt5Login;
