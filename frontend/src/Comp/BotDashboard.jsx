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
  Modal,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalFooter,
  Badge,
} from "@heroui/react";
import {
  PlayIcon,
  CurrencyDollarIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  ClockIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  LinkIcon,
  CogIcon,
  AdjustmentsHorizontalIcon,
  UserCircleIcon,
  ChartBarIcon,
} from "@heroicons/react/24/solid";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import PositionMonitor from "./PositionMonitor";
import SettingsPage from "./Settings";

const BotDashboard = () => {
  const [status, setStatus] = useState({ running: false, paused: false });
  const [mt5Status, setMt5Status] = useState({ connected: false });
  const [data, setData] = useState({});
  const [symbols, setSymbols] = useState([]);
  const [logs, setLogs] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState("XAUUSD");
  const [lotSize, setLotSize] = useState(0.1);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [searchTerm, setSearchTerm] = useState("");
  const [positions, setPositions] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [historicalData, setHistoricalData] = useState([]);
  const [symbolConfigs, setSymbolConfigs] = useState({});
  const [performanceReport, setPerformanceReport] = useState(null);
  const [isConfigModalOpen, setIsConfigModalOpen] = useState(false);
  const [currentConfig, setCurrentConfig] = useState({
    symbol: "",
    strategy: "",
    emotion: "",
    max_positions: 3,
    custom_lot_size: null,
    custom_timeframe: null,
  });
  const apiUrl = "http://localhost:8000";

  // Fetch initial data and set up polling
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [symbolsRes, mt5StatusRes, configsRes, reportRes] =
          await Promise.all([
            axios.get(`${apiUrl}/symbols`),
            axios.get(`${apiUrl}/mt5_status`),
            axios.get(`${apiUrl}/symbol_configs`),
            axios.get(`${apiUrl}/performance_report`),
          ]);
        setSymbols(symbolsRes.data);
        setMt5Status(mt5StatusRes.data);
        setSymbolConfigs(configsRes.data);
        setPerformanceReport(reportRes.data);
      } catch (error) {
        setMessage(`Error fetching initial data: ${error.message}`);
      }
    };
    fetchInitialData();

    const interval = setInterval(async () => {
      try {
        const [
          statusRes,
          dataRes,
          logsRes,
          mt5StatusRes,
          positionsRes,
          historyRes,
          configsRes,
          reportRes,
        ] = await Promise.all([
          axios.get(`${apiUrl}/status`),
          axios.get(`${apiUrl}/data`),
          axios.get(`${apiUrl}/logs`),
          axios.get(`${apiUrl}/mt5_status`),
          axios.get(`${apiUrl}/open_positions`),
          axios.get(`${apiUrl}/trade_history`),
          axios.get(`${apiUrl}/symbol_configs`),
          axios.get(`${apiUrl}/performance_report`),
        ]);
        setStatus(statusRes.data);
        setData(dataRes.data);
        setLogs(logsRes.data.logs || []);
        setMt5Status(mt5StatusRes.data);
        setTradeHistory(historyRes.data);
        setHistoricalData([historicalData.data]);
        setPositions(positionsRes.data.positions || []);
        setSymbolConfigs(configsRes.data);
        setPerformanceReport(reportRes.data);
      } catch (error) {
        setMessage(`Error fetching data: ${error.message}`);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [apiUrl, historicalData]);

  // Generic action handler
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
      // Refresh relevant data after action
      if (
        endpoint.includes("symbol") ||
        endpoint.includes("emotion") ||
        endpoint.includes("strategy")
      ) {
        const configsRes = await axios.get(`${apiUrl}/symbol_configs`);
        setSymbolConfigs(configsRes.data);
      }
      return response.data;
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
      throw error;
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
  const handleClosePosition = (ticket) =>
    handleAction(`Close Position ${ticket}`, `close_position/${ticket}`);
  const handleAdjustEmotions = () =>
    handleAction("Adjust Emotions", "adjust_emotions_based_on_performance");
  const handleResetStats = () =>
    handleAction("Reset Stats", "reset_daily_stats");
  const handleSetupDefaultSymbols = () =>
    handleAction("Setup Default Symbols", "setup_default_symbols");

  // New symbol config handlers
  const openConfigModal = (symbol = null) => {
    if (symbol && symbolConfigs[symbol]) {
      setCurrentConfig({
        symbol,
        strategy: symbolConfigs[symbol].strategy,
        emotion: symbolConfigs[symbol].emotion,
        max_positions: symbolConfigs[symbol].max_positions,
        custom_lot_size: symbolConfigs[symbol].custom_lot_size,
        custom_timeframe: symbolConfigs[symbol].custom_timeframe,
      });
    } else {
      setCurrentConfig({
        symbol: selectedSymbol,
        strategy: "balanced",
        emotion: "standard",
        max_positions: 3,
        custom_lot_size: null,
        custom_timeframe: null,
      });
    }
    setIsConfigModalOpen(true);
  };


  const handleToggleSymbol = async (symbol, enabled) => {
    await handleAction(enabled ? "Enable" : "Disable", "toggle_symbol", {
      symbol,
      enabled,
    });
  };

  const handleUpdateEmotion = async (symbol, emotion) => {
    await handleAction("Update Emotion", "update_emotion", { symbol, emotion });
  };

  const handleUpdateStrategy = async (symbol, strategy) => {
    await handleAction("Update Strategy", "update_strategy", {
      symbol,
      strategy,
    });
  };

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

      {/* Tabs */}
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
              {/* MT5 Connection Status */}
              <div className="mb-6">
                <div className="bg-gray-800 text-white p-4 rounded-xl shadow-lg">
                  <h3 className="text-xl font-semibold mb-2 flex items-center space-x-2">
                    <LinkIcon className="w-6 h-6 text-blue-400" />
                    <span>MT5 Connection</span>
                  </h3>
                  {mt5Status.connected ? (
                    <div className="space-y-2">
                      <span className="bg-green-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                        Connected
                      </span>
                      <p className="text-sm text-gray-400">
                        Account: {mt5Status.account_info?.login || "N/A"}
                      </p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <span className="bg-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                        Disconnected
                      </span>
                      <Link
                        to="/mt5-login"
                        className="text-blue-600 hover:underline flex items-center space-x-1"
                      >
                        <LinkIcon className="w-5 h-5" />
                        <span>Connect to MT5</span>
                      </Link>
                      {mt5Status.last_error && (
                        <p className="text-sm text-red-400">
                          Error: {mt5Status.last_error}
                        </p>
                      )}
                    </div>
                  )}
                </div>
              </div>

              {/* Bot Controls */}
              <div className="mb-8">
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3 sm:gap-4">
                  <Button
                    onPress={handleStart}
                    isDisabled={
                      status.running || loading || !mt5Status.connected
                    }
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
                    isDisabled={
                      !status.running || loading || !mt5Status.connected
                    }
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
                    isDisabled={
                      !status.running || loading || !mt5Status.connected
                    }
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
                    isDisabled={loading || !mt5Status.connected}
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
                    isDisabled={loading || !mt5Status.connected}
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
                      isDisabled={loading || !mt5Status.connected}
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
                      isDisabled={loading || !mt5Status.connected}
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
                      isDisabled={loading || !mt5Status.connected}
                      aria-label="Enter lot size"
                      placeholder="Enter lot size"
                    />
                    <Button
                      onPress={handleLotSizeChange}
                      radius="full"
                      variant="ghost"
                      color="primary"
                      isDisabled={loading || !mt5Status.connected}
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
                      isDisabled={
                        loading ||
                        !mt5Status.connected ||
                        !status.suggested_lot_size
                      }
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

              {/* Strategy Management Section */}
              <div className="mb-8">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-2xl font-bold text-gray-900 tracking-tight">
                    Strategy Management
                  </h2>
                  <div className="flex space-x-2">
                    <Button
                      onPress={handleAdjustEmotions}
                      color="secondary"
                      variant="ghost"
                      size="sm"
                      startContent={
                        <AdjustmentsHorizontalIcon className="w-5 h-5" />
                      }
                    >
                      Auto-Adjust Emotions
                    </Button>
                    <Button
                      onPress={handleResetStats}
                      color="warning"
                      variant="ghost"
                      size="sm"
                      startContent={<ClockIcon className="w-5 h-5" />}
                    >
                      Reset Daily Stats
                    </Button>
                    <Button
                      onPress={handleSetupDefaultSymbols}
                      color="primary"
                      variant="ghost"
                      size="sm"
                      startContent={<CogIcon className="w-5 h-5" />}
                    >
                      Setup Defaults
                    </Button>
                    <Button
                      onPress={() => openConfigModal()}
                      color="success"
                      variant="ghost"
                      size="sm"
                      startContent={<UserCircleIcon className="w-5 h-5" />}
                    >
                      Add Symbol Config
                    </Button>
                  </div>
                </div>

                {/* Symbol Configurations Table */}
                <Table
                  aria-label="Symbol configurations table"
                  className="mb-8"
                >
                  <TableHeader>
                    <TableColumn>Symbol</TableColumn>
                    <TableColumn>Status</TableColumn>
                    <TableColumn>Strategy</TableColumn>
                    <TableColumn>Emotion</TableColumn>
                    <TableColumn>Positions</TableColumn>
                    <TableColumn>Daily P&L</TableColumn>
                    <TableColumn>Actions</TableColumn>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(symbolConfigs).map(([symbol, config]) => (
                      <TableRow key={symbol}>
                        <TableCell>{symbol}</TableCell>
                        <TableCell>
                          <Badge
                            color={config.enabled ? "success" : "danger"}
                            variant="flat"
                          >
                            {config.enabled ? "Enabled" : "Disabled"}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge
                            color={
                              config.strategy === "aggressive"
                                ? "danger"
                                : config.strategy === "passive"
                                ? "primary"
                                : "success"
                            }
                            variant="flat"
                          >
                            {config.strategy}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          <Badge
                            color={
                              config.emotion === "greedy"
                                ? "danger"
                                : config.emotion === "fearful"
                                ? "warning"
                                : "success"
                            }
                            variant="flat"
                          >
                            {config.emotion}
                          </Badge>
                        </TableCell>
                        <TableCell>
                          {performanceReport?.symbol_performance[symbol]
                            ?.current_positions || 0}
                          /{config.max_positions}
                        </TableCell>
                        <TableCell
                          className={
                            config.daily_pnl >= 0
                              ? "text-green-600"
                              : "text-red-600"
                          }
                        >
                          ${config.daily_pnl?.toFixed(2) || "0.00"}
                        </TableCell>
                        <TableCell>
                          <div className="flex space-x-2">
                            <Button
                              size="sm"
                              variant="flat"
                              color="primary"
                              onPress={() => openConfigModal(symbol)}
                            >
                              Edit
                            </Button>
                            <Button
                              size="sm"
                              variant="flat"
                              color={config.enabled ? "danger" : "success"}
                              onPress={() =>
                                handleToggleSymbol(symbol, !config.enabled)
                              }
                            >
                              {config.enabled ? "Disable" : "Enable"}
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>

                {/* Performance Summary */}
                {performanceReport && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                    <Card className="p-4 bg-blue-50">
                      <CardHeader className="flex items-center space-x-2">
                        <ChartBarIcon className="w-6 h-6 text-blue-600" />
                        <h3 className="text-lg font-semibold text-black">Global Stats</h3>
                      </CardHeader>
                      <CardBody>
                        <div className="space-y-2 text-black">
                          <p>
                            Active Symbols:{" "}
                            {performanceReport.global_stats.active_symbols}
                          </p>
                          <p>
                            Total Positions:{" "}
                            {performanceReport.global_stats.total_positions}
                          </p>
                          <p>
                            Daily Profit: $
                            {performanceReport.global_stats.daily_profit?.toFixed(2)}
                          </p>
                          <p>
                            Total Trades:{" "}
                            {performanceReport.global_stats.total_trades}
                          </p>
                        </div>
                      </CardBody>
                    </Card>
                    <Card className="p-4 bg-purple-50">
                      <CardHeader className="flex items-center space-x-2">
                        <AdjustmentsHorizontalIcon className="w-6 h-6 text-purple-600" />
                        <h3 className="text-lg font-semibold text-black">
                          Strategy Distribution
                        </h3>
                      </CardHeader>
                      <CardBody>
                        <div className="space-y-2 text-black">
                          {Object.entries(
                            performanceReport.strategy_distribution
                          ).map(([strategy, count]) => (
                            <div
                              key={strategy}
                              className="flex justify-between"
                            >
                              <span className="capitalize">{strategy}</span>
                              <span>{count}</span>
                            </div>
                          ))}
                        </div>
                      </CardBody>
                    </Card>
                    <Card className="p-4 bg-green-50">
                      <CardHeader className="flex items-center space-x-2">
                        <UserCircleIcon className="w-6 h-6 text-green-600" />
                        <h3 className="text-lg font-semibold text-black">
                          Emotion Distribution
                        </h3>
                      </CardHeader>
                      <CardBody>
                        <div className="space-y-2 text-black">
                          {Object.entries(
                            performanceReport.emotion_distribution
                          ).map(([emotion, count]) => (
                            <div key={emotion} className="flex justify-between">
                              <span className="capitalize">{emotion}</span>
                              <span>{count}</span>
                            </div>
                          ))}
                        </div>
                      </CardBody>
                    </Card>
                  </div>
                )}
              </div>
              <div className="mb-8 px-4">
                <h2 className="text-2xl font-bold text-gray-900 mb-6 tracking-tight">
                  Open Positions
                </h2>
                {positions.length === 0 ? (
                  <p className="text-gray-800">No open positions</p>
                ) : (
                  <Table aria-label="Open positions table">
                    <TableHeader>
                      <TableColumn>Symbol</TableColumn>
                      <TableColumn>Type</TableColumn>
                      <TableColumn>Volume</TableColumn>
                      <TableColumn>Entry Price</TableColumn>
                      <TableColumn>Current Price</TableColumn>
                      <TableColumn>Profit</TableColumn>
                      <TableColumn>Action</TableColumn>
                    </TableHeader>
                    <TableBody>
                      {positions.map((pos) => (
                        <TableRow key={pos.ticket}>
                          <TableCell>{pos.symbol}</TableCell>
                          <TableCell>{pos.type}</TableCell>
                          <TableCell>{pos.volume.toFixed(2)}</TableCell>
                          <TableCell>{pos.entry_price.toFixed(5)}</TableCell>
                          <TableCell>{pos.current_price.toFixed(5)}</TableCell>
                          <TableCell
                            className={
                              pos.profit >= 0
                                ? "text-green-600"
                                : "text-red-600"
                            }
                          >
                            {pos.profit.toFixed(2)} ({pos.profit_pct.toFixed(2)}
                            %)
                          </TableCell>
                          <TableCell>
                            <Button
                              onPress={() => handleClosePosition(pos.ticket)}
                              isDisabled={loading || !mt5Status.connected}
                              color="danger"
                              size="sm"
                              variant="ghost"
                              aria-label={`Close position ${pos.ticket}`}
                            >
                              Close
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </div>
              <Accordion>
                <AccordionItem
                  key="trade-history"
                  aria-label="Trade History"
                  title="Trade History"
                >
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableCell>Ticket</TableCell>
                        <TableCell>Symbol</TableCell>
                        <TableCell>Action</TableCell>
                        <TableCell>Volume</TableCell>
                        <TableCell>Entry Price</TableCell>
                        <TableCell>Exit Price</TableCell>
                        <TableCell>Profit</TableCell>
                        <TableCell>Open Time</TableCell>
                        <TableCell>Close Time</TableCell>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {tradeHistory.map((trade) => (
                        <TableRow key={trade.ticket}>
                          <TableCell>{trade.ticket}</TableCell>
                          <TableCell>{trade.symbol}</TableCell>
                          <TableCell>{trade.action}</TableCell>
                          <TableCell>{trade.volume}</TableCell>
                          <TableCell>{trade.entry_price}</TableCell>
                          <TableCell>{trade.exit_price}</TableCell>
                          <TableCell>{trade.profit}</TableCell>
                          <TableCell>{trade.open_time}</TableCell>
                          <TableCell>{trade.close_time}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </AccordionItem>
              </Accordion>

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
            <Card className="p-6 bg-gray-200 shadow-2xl shadow-indigo-500/50">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900 tracking-tight">
                  Bot Logs
                </h2>
                <div className="flex space-x-2">
                  <Input
                    type="text"
                    placeholder="Search logs..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-64"
                    size="sm"
                    variant="bordered"
                    color="default"
                    aria-label="Search logs"
                  />
                  <Button
                    onPress={clearLogs}
                    color="danger"
                    variant="ghost"
                    size="sm"
                    aria-label="Clear logs"
                  >
                    Clear Logs
                  </Button>
                </div>
              </div>
              <div className="bg-gray-800 rounded-lg p-4 h-96 overflow-y-auto">
                {filteredLogs.length === 0 ? (
                  <p className="text-gray-400">No logs available</p>
                ) : (
                  <div className="space-y-1 font-mono text-sm">
                    {filteredLogs.map((log, index) => (
                      <div
                        key={index}
                        className={`p-2 rounded ${
                          log.toLowerCase().includes("error")
                            ? "bg-red-900 text-red-100"
                            : log.toLowerCase().includes("warning")
                            ? "bg-yellow-900 text-yellow-100"
                            : "bg-gray-700 text-gray-300"
                        }`}
                      >
                        {log}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </Card>
          </motion.div>
        </Tab>

        {/* Position Monitor Tab */}
        <Tab key="monitor" title="Position Monitor">
          <PositionMonitor
            positions={positions}
            onClosePosition={handleClosePosition}
            loading={loading}
            mt5Status={mt5Status}
          />
        </Tab>

        {/* Settings Tab */}
        <Tab key="settings" title="Settings">
          <SettingsPage
            symbols={symbols}
            symbolConfigs={symbolConfigs}
            onToggleSymbol={handleToggleSymbol}
            onUpdateEmotion={handleUpdateEmotion}
            onUpdateStrategy={handleUpdateStrategy}
          />
        </Tab>
      </Tabs>

      {/* Symbol Configuration Modal */}
      <Modal
        isOpen={isConfigModalOpen}
        onClose={() => setIsConfigModalOpen(false)}
        size="lg"
      >
        <ModalContent>
          <ModalHeader className="flex flex-col gap-1">
            {currentConfig.symbol
              ? `Configure ${currentConfig.symbol}`
              : "Add New Symbol Configuration"}
          </ModalHeader>
          <ModalBody>
            <div className="space-y-4">
              {!currentConfig.symbol && (
                <Select
                  label="Symbol"
                  selectedKeys={[currentConfig.symbol]}
                  onSelectionChange={(keys) =>
                    setCurrentConfig({
                      ...currentConfig,
                      symbol: Array.from(keys)[0],
                    })
                  }
                  className="w-full"
                >
                  {symbols.map((sym) => (
                    <SelectItem key={sym} value={sym}>
                      {sym}
                    </SelectItem>
                  ))}
                </Select>
              )}

              <Select
                label="Strategy"
                selectedKeys={[currentConfig.strategy]}
                onSelectionChange={(keys) =>
                  setCurrentConfig({
                    ...currentConfig,
                    strategy: Array.from(keys)[0],
                  })
                }
                className="w-full"
              >
                <SelectItem key="aggressive" value="aggressive">
                  Aggressive
                </SelectItem>
                <SelectItem key="balanced" value="balanced">
                  Balanced
                </SelectItem>
                <SelectItem key="passive" value="passive">
                  Passive
                </SelectItem>
              </Select>

              <Select
                label="Emotion"
                selectedKeys={[currentConfig.emotion]}
                onSelectionChange={(keys) =>
                  setCurrentConfig({
                    ...currentConfig,
                    emotion: Array.from(keys)[0],
                  })
                }
                className="w-full"
              >
                <SelectItem key="greedy" value="greedy">
                  Greedy
                </SelectItem>
                <SelectItem key="standard" value="standard">
                  Standard
                </SelectItem>
                <SelectItem key="fearful" value="fearful">
                  Fearful
                </SelectItem>
              </Select>

              <Input
                label="Max Positions"
                type="number"
                min="1"
                max="10"
                value={currentConfig.max_positions}
                onChange={(e) =>
                  setCurrentConfig({
                    ...currentConfig,
                    max_positions: parseInt(e.target.value),
                  })
                }
              />

              <Input
                label="Custom Lot Size (optional)"
                type="number"
                step="0.01"
                min="0"
                value={currentConfig.custom_lot_size || ""}
                onChange={(e) =>
                  setCurrentConfig({
                    ...currentConfig,
                    custom_lot_size: parseFloat(e.target.value),
                  })
                }
              />

              <Input
                label="Custom Timeframe (optional)"
                placeholder="e.g., H1, M15"
                value={currentConfig.custom_timeframe || ""}
                onChange={(e) =>
                  setCurrentConfig({
                    ...currentConfig,
                    custom_timeframe: e.target.value,
                  })
                }
              />
            </div>
          </ModalBody>
          <ModalFooter>
            <Button
              color="danger"
              variant="light"
              onPress={() => setIsConfigModalOpen(false)}
            >
              Cancel
            </Button>
            <Button
              color="primary"
              isDisabled={!currentConfig.symbol}
            >
              Save Configuration
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
};

export default BotDashboard;
