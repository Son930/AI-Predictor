import { useState, useEffect } from "react";
import axios from "axios";
import "../../src/App.css";
import {
  Button,
  Card,
  CardBody,
  CardHeader,
  Tabs,
  Tab,
  Input,
  Select,
  SelectItem,
  Table,
  TableHeader,
  TableColumn,
  TableBody,
  TableRow,
  TableCell,
  Alert,
  Accordion,
  AccordionItem,
  Spinner,
} from "@heroui/react";
import {
  PlayIcon,
  CurrencyDollarIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
} from "@heroicons/react/24/solid";
import { motion } from "framer-motion";
import PositionMonitor from "./PositionMonitor";

const BotDashboard = () => {
  const [status, setStatus] = useState({ running: false, paused: false });
  const [data, setData] = useState({});
  const [symbols, setSymbols] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState("XAUUSD");
  const [lotSize, setLotSize] = useState(0.1);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [searchTerm, setSearchTerm] = useState(""); // Added for log filtering
  const apiUrl = "http://localhost:8000";

  // Fetch initial data and set up polling
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const symbolsRes = await axios.get(`${apiUrl}/symbols`);
        setSymbols(symbolsRes.data);
      } catch (error) {
        setMessage(`Error fetching symbols: ${error.message}`);
      }
    };
    fetchInitialData();

    const interval = setInterval(async () => {
      try {
        const [statusRes, dataRes, logsRes] = await Promise.all([
          axios.get(`${apiUrl}/status`),
          axios.get(`${apiUrl}/data`),
          axios.get(`${apiUrl}/logs`),
        ]);
        setStatus(statusRes.data);
        setData(dataRes.data);
        setLogs(logsRes.data.logs || []);
      } catch (error) {
        setMessage(`Error fetching data: ${error.message}`);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [apiUrl]);

  // Generic action handler with improved error handling
  const handleAction = async (action, endpoint, data = null) => {
    setLoading(true);
    setMessage("");
    try {
      const response = data
        ? await axios.post(`${apiUrl}/${endpoint}`, data, {
            headers: { "Content-Type": "application/json" },
          })
        : await axios.post(`${apiUrl}/${endpoint}`);
      setMessage(response.data.message || `${action} successful`);
    } catch (error) {
      if (error.response) {
        setMessage(
          `Error: ${error.response.data.error || "Unknown server error"}`
        );
      } else if (error.request) {
        setMessage("Error: No response from server");
      } else {
        setMessage(`Error: ${error.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  // Action handlers
  const clearLogs = () => setLogs([]);
  const handleStart = () => handleAction("Start", "start");
  const handleStop = () => handleAction("Stop", "stop");
  const handlePause = () =>
    handleAction(status.paused ? "Resume" : "Pause", "pause");
  const handleSymbolChange = () =>
    handleAction("Set Symbol", "set_symbol", { symbol: selectedSymbol });
  const handleLotSizeChange = () =>
    handleAction("Set Lot Size", "set_lot_size", { lot_size: lotSize });
  const handleUseSuggested = () =>
    handleAction("Use Suggested", "use_suggested_lot_size");
  const handleManualTrade = (action) =>
    handleAction(`Manual ${action}`, "manual_trade", {
      action,
      lot_size: lotSize,
    });

  // Filter logs based on search term
  const filteredLogs = logs.filter((log) =>
    log.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="max-w-7xl mx-auto h-full relative p-4">
      {/* Dynamic Alert */}
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

      {/* Enhanced Tabs */}
      <Tabs
        selectedKey={activeTab}
        onSelectionChange={setActiveTab}
        className="mb-8"
        color="primary"
        variant="underlined"
        size="lg"
      >
        {/* Dashboard Tab */}
        <Tab key="dashboard" title="Dashboard">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            {loading && (
              <div className="absolute inset-0 flex justify-center items-center bg-gray-200 bg-opacity-50">
                <Spinner size="lg" />
              </div>
            )}
            <Card className="p-6 bg-gray-200 shadow-2xl shadow-indigo-500/50 relative">
              {/* Bot Controls */}
              <div className="mb-8">
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 sm:gap-4">
                  <Button
                    onPress={handleStart}
                    isDisabled={status.running || loading}
                    color="primary"
                    radius="full"
                    variant="ghost"
                    className="flex items-center justify-center space-x-2 min-h-[44px] w-full"
                    aria-label="Start trading bot"
                    title="Start the trading bot"
                  >
                    <PlayIcon className="w-5 h-5" />
                    <span>Start Bot</span>
                  </Button>
                  <Button
                    onPress={handleStop}
                    isDisabled={!status.running || loading}
                    color="danger"
                    variant="ghost"
                    radius="full"
                    className="flex items-center justify-center space-x-2 min-h-[44px] w-full"
                    aria-label="Stop trading bot"
                    title="Stop the trading bot"
                  >
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 10h6v4H9v-4z"
                      />
                    </svg>
                    <span>Stop Bot</span>
                  </Button>
                  <Button
                    onPress={handlePause}
                    isDisabled={!status.running || loading}
                    color="warning"
                    radius="full"
                    variant="ghost"
                    className="flex items-center justify-center space-x-2 min-h-[44px] w-full"
                    aria-label={
                      status.paused ? "Resume trading bot" : "Pause trading bot"
                    }
                    title={status.paused ? "Resume the bot" : "Pause the bot"}
                  >
                    <svg
                      className="w-5 h-5"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      {status.paused ? (
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M14.752 11.168l-3.197-2.2A1 1 0 0010 9.768v4.464a1 1 0 001.555.832l3.197-2.2a1 1 0 000-1.664zM21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      ) : (
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"
                        />
                      )}
                    </svg>
                    <span>{status.paused ? "Resume" : "Pause"}</span>
                  </Button>
                  <Button
                    onPress={() => handleManualTrade("buy")}
                    isDisabled={loading}
                    color="success"
                    radius="full"
                    variant="ghost"
                    className="flex items-center justify-center space-x-2 min-h-[44px] w-full"
                    aria-label="Execute manual buy trade"
                    title="Execute a manual buy trade"
                  >
                    <ArrowUpIcon className="w-5 h-5" />
                    <span>Manual Buy</span>
                  </Button>
                  <Button
                    onPress={() => handleManualTrade("sell")}
                    isDisabled={loading}
                    color="danger"
                    radius="full"
                    variant="ghost"
                    className="flex items-center justify-center space-x-2 min-h-[44px] w-full"
                    aria-label="Execute manual sell trade"
                    title="Execute a manual sell trade"
                  >
                    <ArrowDownIcon className="w-5 h-5" />
                    <span>Manual Sell</span>
                  </Button>
                </div>
              </div>

              {/* Symbol and Lot Size Controls */}
              <div className="grid grid-cols-1 gap-4 sm:gap-6 mb-8 md:grid-cols-2">
                <div className="space-y-2">
                  <label
                    htmlFor="symbol-select"
                    className="block text-sm sm:text-base font-medium text-gray-800"
                  >
                    Choose Symbol
                  </label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <Select
                      id="symbol-select"
                      selectedKeys={[selectedSymbol]}
                      onSelectionChange={(keys) =>
                        setSelectedSymbol(Array.from(keys)[0])
                      }
                      aria-label="Select trading symbol"
                      className="w-full sm:flex-1 text-gray-800"
                      color="default"
                      variant="bordered"
                      placeholder="Select a symbol"
                      size="sm"
                      isDisabled={loading}
                    >
                      {symbols.map((sym) => (
                        <SelectItem
                          key={sym}
                          value={sym}
                          className="text-gray-800"
                        >
                          {sym}
                        </SelectItem>
                      ))}
                    </Select>
                    <Button
                      onPress={handleSymbolChange}
                      radius="full"
                      variant="ghost"
                      color="primary"
                      isDisabled={loading}
                      className="w-full sm:w-auto min-w-[100px]"
                      aria-label="Confirm selected symbol"
                      title="Set the selected symbol"
                    >
                      Set Symbol
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <label
                    htmlFor="lot-size-input"
                    className="block text-sm sm:text-base font-medium text-gray-800"
                  >
                    Lot Size
                  </label>
                  <div className="flex flex-col sm:flex-row gap-2">
                    <Input
                      id="lot-size-input"
                      type="number"
                      step="0.01"
                      min="0"
                      value={lotSize.toString()}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value) && value >= 0) setLotSize(value);
                      }}
                      className="w-full sm:flex-1"
                      size="sm"
                      variant="bordered"
                      color="default"
                      isDisabled={loading}
                      aria-label="Enter lot size"
                      placeholder="Enter lot size"
                    />
                    <Button
                      onPress={handleLotSizeChange}
                      radius="full"
                      variant="ghost"
                      color="primary"
                      isDisabled={loading}
                      className="w-full sm:w-auto min-w-[100px]"
                      aria-label="Confirm lot size"
                      title="Set the lot size"
                    >
                      Set Lot Size
                    </Button>
                    <Button
                      onPress={handleUseSuggested}
                      radius="full"
                      variant="ghost"
                      color="success"
                      isDisabled={loading || !status.suggested_lot_size}
                      className="w-full sm:w-auto min-w-[120px]"
                      aria-label="Use suggested lot size"
                      title="Use the suggested lot size"
                    >
                      Suggested (
                      {status.suggested_lot_size?.toFixed(2) ?? "N/A"})
                    </Button>
                  </div>
                </div>
              </div>

              {/* Status Section */}
              <div className="mb-8 px-4">
                <h2 className="text-2xl font-bold text-gray-900 mb-6 tracking-tight">
                  Status
                </h2>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <motion.div
                    key={`${status.running}-${status.paused}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <div className="bg-gray-800 text-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
                      <h3 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                        <PlayIcon className="w-6 h-6 text-blue-400" />
                        <span>Bot Status</span>
                      </h3>
                      <div className="space-y-4">
                        <div className="flex items-center space-x-2">
                          {status.running ? (
                            <span className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                              Running
                            </span>
                          ) : (
                            <span className="bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                              Stopped
                            </span>
                          )}
                        </div>
                        <div className="flex items-center space-x-2">
                          {status.paused ? (
                            <span className="bg-yellow-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                              Paused
                            </span>
                          ) : (
                            <span className="bg-gray-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                              Active
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </motion.div>
                  <motion.div
                    key={`${status.symbol}-${status.last_prediction}-${status.last_confidence}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <div className="bg-gray-800 text-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
                      <h3 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                        <CurrencyDollarIcon className="w-6 h-6 text-green-400" />
                        <span>Trading Info</span>
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <p className="text-sm text-gray-400">Symbol</p>
                          <p className="text-lg font-semibold">
                            {status.symbol || "N/A"}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-400">
                            Last Prediction
                          </p>
                          <p className="text-lg font-semibold flex items-center space-x-2">
                            {status.last_prediction === 1 ? (
                              <ArrowUpIcon className="w-5 h-5 text-green-400" />
                            ) : status.last_prediction === -1 ? (
                              <ArrowDownIcon className="w-5 h-5 text-red-400" />
                            ) : (
                              <ClockIcon className="w-5 h-5 text-yellow-400" />
                            )}
                            <span>
                              {status.last_prediction === 1
                                ? "Buy"
                                : status.last_prediction === -1
                                ? "Sell"
                                : "Hold"}
                            </span>
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-400">
                            Last Confidence
                          </p>
                          <p
                            className={`text-lg font-semibold ${
                              status.last_confidence > 0.8
                                ? "text-green-400"
                                : status.last_confidence > 0.5
                                ? "text-yellow-400"
                                : "text-red-400"
                            }`}
                          >
                            {status.last_confidence?.toFixed(2) || "N/A"}
                          </p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                  <motion.div
                    key={`${status.lot_size}-${status.error_count}-${status.last_error}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <div className="bg-gray-800 text-white p-6 rounded-xl shadow-lg hover:shadow-xl transition-shadow">
                      <h3 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                        {status.error_count > 0 ? (
                          <ExclamationTriangleIcon className="w-6 h-6 text-yellow-400" />
                        ) : (
                          <InformationCircleIcon className="w-6 h-6 text-blue-400" />
                        )}
                        <span>Additional Status</span>
                      </h3>
                      <div className="space-y-4">
                        <div>
                          <p className="text-sm text-gray-400">Lot Size</p>
                          <p className="text-lg font-semibold">
                            {status.lot_size || "N/A"}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-400">Error Count</p>
                          <p
                            className={`text-lg font-semibold ${
                              status.error_count > 0
                                ? "text-red-400"
                                : "text-green-400"
                            }`}
                          >
                            {status.error_count || 0}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-gray-400">Last Error</p>
                          <p
                            className={`text-lg font-semibold ${
                              status.last_error
                                ? "text-red-400"
                                : "text-gray-400"
                            }`}
                          >
                            {status.last_error || "None"}
                          </p>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                </div>
              </div>

              {/* Latest Data Section */}
              <Accordion>
                <AccordionItem
                  key="1"
                  aria-label="Latest Data"
                  title="Latest Data"
                  className="text-xl font-semibold text-gray-900 mb-4"
                >
                  <Card className="p-4">
                    <CardBody>
                      <Table>
                        <TableHeader>
                          <TableColumn>Key</TableColumn>
                          <TableColumn>Value</TableColumn>
                        </TableHeader>
                        <TableBody>
                          {Object.entries(data).map(([key, value]) => (
                            <TableRow key={key}>
                              <TableCell>{key}</TableCell>
                              <TableCell>{JSON.stringify(value)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardBody>
                  </Card>
                </AccordionItem>
              </Accordion>
            </Card>
          </motion.div>
        </Tab>

        {/* Logs Tab */}
        <Tab key="logs" title="Logs">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <Card className="p-6 bg-gradient-to-br from-gray-900 to-gray-800 text-white shadow-2xl rounded-2xl backdrop-blur-md backdrop-opacity-80">
              <CardHeader className="flex justify-between items-center">
                <h2 className="text-2xl font-bold tracking-tight">Logs</h2>
                <div className="flex space-x-2">
                  <Input
                    placeholder="Search logs"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-48"
                    aria-label="Search logs"
                    size透
                    size="sm"
                  />
                  <Button
                    color="secondary"
                    variant="flat"
                    size="sm"
                    onPress={clearLogs}
                    aria-label="Clear all logs"
                  >
                    Clear Logs
                  </Button>
                </div>
              </CardHeader>
              <CardBody>
                <div className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-gray-800">
                  <ul
                    className="space-y-3"
                    role="list"
                    aria-label="Log entries"
                  >
                    {filteredLogs.map((log, index) => (
                      <li
                        key={index}
                        className="bg-gray-800 bg-opacity-50 p-3 rounded-lg hover:bg-opacity-70 transition-all duration-200"
                        role="listitem"
                      >
                        <div className="flex items-start space-x-3">
                          <span className="text-xs text-gray-400 flex-shrink-0">
                            {new Date().toLocaleTimeString()}
                          </span>
                          <p className="text-sm">{log.trim()}</p>
                        </div>
                      </li>
                    ))}
                  </ul>
                </div>
              </CardBody>
            </Card>
          </motion.div>
        </Tab>

        {/* Positions Tab */}
        <Tab key="positions" title="Positions">
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.3 }}
          >
            <PositionMonitor />
          </motion.div>
        </Tab>
      </Tabs>
    </div>
  );
};

export default BotDashboard;
