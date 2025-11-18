/* eslint-disable no-unused-vars */
// Redesigned BotDashboard.jsx
import { useState, useEffect } from "react";
import axios from "axios";
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
    <div className="max-w-7xl mx-auto h-full relative p-4 bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100">
      {/* Dynamic Alert */}
      {message && (
        <Alert
          description={message}
          title={message.startsWith("Error:") ? "Error" : "Success"}
          radius="full"
          className={`mb-4 p-4 ${
            message.startsWith("Error:")
              ? "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300"
              : "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300"
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
              <div className="absolute inset-0 flex justify-center items-center bg-gray-200 dark:bg-gray-800 bg-opacity-50 dark:bg-opacity-50">
                <Spinner size="lg" />
              </div>
            )}
            <Card className="p-6 bg-white dark:bg-gray-800 shadow-xl rounded-xl">
              {/* MT5 Connection Status */}
              <div className="mb-6">
                <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-xl shadow-md">
                  <h3 className="text-xl font-semibold mb-2 flex items-center">
                    MT5 Connection Status
                  </h3>
                  <Badge color={mt5Status.connected ? "success" : "danger"}>
                    {mt5Status.connected ? "Connected" : "Disconnected"}
                  </Badge>
                </div>
              </div>

              {/* Latest Data Section */}
              <Accordion>
                <AccordionItem
                  key="1"
                  aria-label="Latest Data"
                  title="Latest Data"
                  className="text-xl font-semibold mb-4"
                >
                  <Card className="p-4 bg-white dark:bg-gray-800">
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
            <Card className="p-6 bg-white dark:bg-gray-800 shadow-xl rounded-xl">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold tracking-tight">Bot Logs</h2>
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
              <div className="bg-gray-100 dark:bg-gray-700 rounded-lg p-4 h-96 overflow-y-auto">
                {filteredLogs.length === 0 ? (
                  <p className="text-gray-500 dark:text-gray-400">
                    No logs available
                  </p>
                ) : (
                  <div className="space-y-1 font-mono text-sm">
                    {filteredLogs.map((log, index) => (
                      <div
                        key={index}
                        className={`p-2 rounded ${
                          log.toLowerCase().includes("error")
                            ? "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300"
                            : log.toLowerCase().includes("warning")
                            ? "bg-yellow-100 dark:bg-yellow-900 text-yellow-700 dark:text-yellow-300"
                            : "bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300"
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
        <ModalContent className="bg-white dark:bg-gray-800">
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
            <Button color="primary" isDisabled={!currentConfig.symbol}>
              Save Configuration
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </div>
  );
};

export default BotDashboard;
