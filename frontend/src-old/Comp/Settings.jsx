// Redesigned Settings.jsx
/* eslint-disable no-unused-vars */
import { useState, useEffect } from "react";
import {
  Card,
  CardHeader,
  CardBody,
  Input,
  Button,
  Switch,
  Select,
  SelectItem,
  Divider,
  Chip,
  Slider,
} from "@heroui/react";
import {
  CogIcon,
  BellIcon,
  ChartBarIcon,
  CurrencyDollarIcon,
  ShieldCheckIcon,
  ClockIcon,
  ArrowPathIcon,
} from "@heroicons/react/24/solid";
import axios from "axios";
import PropTypes from "prop-types";

const defaultSettings = {
  botName: "AI Trading Bot",
  responseDelay: 1,
  enableNotifications: true,
  riskLevel: "medium", // low, medium, high
  maxDailyLossPercent: 5,
  maxConcurrentTrades: 5,
  enableAutoStop: true,
  tradeExecutionMode: "auto", // auto, semi-auto, manual
  enableHedging: false,
  slippageTolerance: 0.5,
  enableTelegramAlerts: false,
  telegramChatId: "",
  enableEmailAlerts: false,
  emailAddress: "",
  enableStopLoss: true,
  defaultStopLossPips: 20,
  enableTakeProfit: true,
  defaultTakeProfitPips: 40,
  enableTrailingStop: false,
  trailingStopDistance: 15,
  enableNewsFilter: true,
  enableWeekendTrading: false,
  enableDebugMode: false,
};

const SettingsPage = ({
  symbols,
  symbolConfigs,
  onToggleSymbol,
  onUpdateEmotion,
  onUpdateStrategy,
  onSaveConfig,
}) => {
  const [settings, setSettings] = useState(defaultSettings);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");
  const [activeTab, setActiveTab] = useState("general");

  useEffect(() => {
    // Load settings from backend when component mounts
    const loadSettings = async () => {
      try {
        setLoading(true);
        const response = await axios.get("/api/settings");
        setSettings(response.data);
      } catch (error) {
        setMessage(`Error loading settings: ${error.message}`);
      } finally {
        setLoading(false);
      }
    };
    loadSettings();
  }, []);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setSettings((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSliderChange = (name, value) => {
    setSettings((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSave = async () => {
    try {
      setLoading(true);
      setMessage("");
      await axios.post("/api/settings", settings);
      setMessage("Settings saved successfully!");
      setTimeout(() => setMessage(""), 3000);
    } catch (error) {
      setMessage(`Error saving settings: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (window.confirm("Are you sure you want to reset to default settings?")) {
      setSettings(defaultSettings);
      setMessage("Settings reset to defaults");
      setTimeout(() => setMessage(""), 3000);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-4">
      <Card className="shadow-lg rounded-xl bg-white dark:bg-gray-800">
        <CardHeader className="border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center space-x-3">
            <CogIcon className="w-8 h-8 text-blue-500 dark:text-blue-400" />
            <h2 className="text-2xl font-bold dark:text-gray-100">
              Bot Settings
            </h2>
          </div>
          <div className="flex space-x-4 mt-4">
            <Button
              variant={activeTab === "general" ? "solid" : "light"}
              color="primary"
              onPress={() => setActiveTab("general")}
              startContent={<CogIcon className="w-5 h-5" />}
            >
              General
            </Button>
            <Button
              variant={activeTab === "risk" ? "solid" : "light"}
              color="danger"
              onPress={() => setActiveTab("risk")}
              startContent={<ShieldCheckIcon className="w-5 h-5" />}
            >
              Risk Management
            </Button>
            <Button
              variant={activeTab === "alerts" ? "solid" : "light"}
              color="warning"
              onPress={() => setActiveTab("alerts")}
              startContent={<BellIcon className="w-5 h-5" />}
            >
              Alerts
            </Button>
            <Button
              variant={activeTab === "symbols" ? "solid" : "light"}
              color="success"
              onPress={() => setActiveTab("symbols")}
              startContent={<CurrencyDollarIcon className="w-5 h-5" />}
            >
              Symbol Settings
            </Button>
          </div>
        </CardHeader>

        <CardBody className="p-6">
          {message && (
            <div
              className={`p-3 mb-6 rounded-md ${
                message.startsWith("Error")
                  ? "bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-300"
                  : "bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300"
              }`}
            >
              {message}
            </div>
          )}

          {loading && (
            <div className="flex justify-center items-center py-8">
              <ArrowPathIcon className="w-8 h-8 animate-spin text-blue-500 dark:text-blue-400" />
            </div>
          )}

          {activeTab === "general" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Input
                  label="Bot Name"
                  name="botName"
                  value={settings.botName}
                  onChange={handleChange}
                  description="The display name for your trading bot"
                />

                <Select
                  label="Trade Execution Mode"
                  name="tradeExecutionMode"
                  selectedKeys={[settings.tradeExecutionMode]}
                  onSelectionChange={(keys) =>
                    handleChange({
                      target: {
                        name: "tradeExecutionMode",
                        value: Array.from(keys)[0],
                      },
                    })
                  }
                >
                  <SelectItem key="auto">Fully Automatic</SelectItem>
                  <SelectItem key="semi-auto">
                    Semi-Automatic (Confirm trades)
                  </SelectItem>
                  <SelectItem key="manual">Manual (Signals only)</SelectItem>
                </Select>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium mb-1 dark:text-gray-300">
                    Response Delay: {settings.responseDelay} second(s)
                  </label>
                  <Slider
                    name="responseDelay"
                    value={settings.responseDelay}
                    minValue={0}
                    maxValue={10}
                    step={0.5}
                    onChange={(value) =>
                      handleSliderChange("responseDelay", value)
                    }
                  />
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                    Delay between trade signals and execution
                  </p>
                </div>

                <Switch
                  name="enableHedging"
                  isSelected={settings.enableHedging}
                  onChange={handleChange}
                  className="mt-1"
                >
                  Enable Hedging Mode
                </Switch>
              </div>

              <Divider />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Switch
                  name="enableNewsFilter"
                  isSelected={settings.enableNewsFilter}
                  onChange={handleChange}
                >
                  Filter Trades During High Impact News
                </Switch>

                <Switch
                  name="enableWeekendTrading"
                  isSelected={settings.enableWeekendTrading}
                  onChange={handleChange}
                >
                  Enable Weekend Trading
                </Switch>
              </div>

              <Switch
                name="enableDebugMode"
                isSelected={settings.enableDebugMode}
                onChange={handleChange}
              >
                Enable Debug Mode (Logs more details)
              </Switch>
            </div>
          )}

          {activeTab === "risk" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Select
                  label="Risk Level"
                  name="riskLevel"
                  selectedKeys={[settings.riskLevel]}
                  onSelectionChange={(keys) =>
                    handleChange({
                      target: {
                        name: "riskLevel",
                        value: Array.from(keys)[0],
                      },
                    })
                  }
                >
                  <SelectItem key="low">Low</SelectItem>
                  <SelectItem key="medium">Medium</SelectItem>
                  <SelectItem key="high">High</SelectItem>
                </Select>

                <Input
                  label="Max Daily Loss (%)"
                  type="number"
                  name="maxDailyLossPercent"
                  min="1"
                  max="20"
                  value={settings.maxDailyLossPercent}
                  onChange={handleChange}
                />
              </div>

              <div>
                <label className="block text-sm font-medium mb-1 dark:text-gray-300">
                  Max Concurrent Trades: {settings.maxConcurrentTrades}
                </label>
                <Slider
                  name="maxConcurrentTrades"
                  value={settings.maxConcurrentTrades}
                  minValue={1}
                  maxValue={20}
                  onChange={(value) =>
                    handleSliderChange("maxConcurrentTrades", value)
                  }
                />
              </div>

              <Switch
                name="enableAutoStop"
                isSelected={settings.enableAutoStop}
                onChange={handleChange}
              >
                Enable Auto Stop on Max Loss
              </Switch>

              <Divider />

              <div className="space-y-4">
                <Switch
                  name="enableStopLoss"
                  isSelected={settings.enableStopLoss}
                  onChange={handleChange}
                >
                  Enable Stop Loss
                </Switch>

                {settings.enableStopLoss && (
                  <Input
                    label="Default Stop Loss (pips)"
                    type="number"
                    name="defaultStopLossPips"
                    min="10"
                    max="100"
                    value={settings.defaultStopLossPips}
                    onChange={handleChange}
                  />
                )}
              </div>

              <div className="space-y-4">
                <Switch
                  name="enableTakeProfit"
                  isSelected={settings.enableTakeProfit}
                  onChange={handleChange}
                >
                  Enable Take Profit
                </Switch>

                {settings.enableTakeProfit && (
                  <Input
                    label="Default Take Profit (pips)"
                    type="number"
                    name="defaultTakeProfitPips"
                    min="10"
                    max="200"
                    value={settings.defaultTakeProfitPips}
                    onChange={handleChange}
                  />
                )}
              </div>

              <div className="space-y-4">
                <Switch
                  name="enableTrailingStop"
                  isSelected={settings.enableTrailingStop}
                  onChange={handleChange}
                >
                  Enable Trailing Stop
                </Switch>

                {settings.enableTrailingStop && (
                  <Input
                    label="Trailing Stop Distance (pips)"
                    type="number"
                    name="trailingStopDistance"
                    min="5"
                    max="50"
                    value={settings.trailingStopDistance}
                    onChange={handleChange}
                  />
                )}
              </div>
            </div>
          )}

          {activeTab === "alerts" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Switch
                  name="enableNotifications"
                  isSelected={settings.enableNotifications}
                  onChange={handleChange}
                >
                  Enable In-App Notifications
                </Switch>

                <Switch
                  name="enableTelegramAlerts"
                  isSelected={settings.enableTelegramAlerts}
                  onChange={handleChange}
                >
                  Enable Telegram Alerts
                </Switch>
              </div>

              {settings.enableTelegramAlerts && (
                <Input
                  label="Telegram Chat ID"
                  name="telegramChatId"
                  value={settings.telegramChatId}
                  onChange={handleChange}
                  description="Your Telegram chat ID for receiving alerts"
                />
              )}

              <Divider />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Switch
                  name="enableEmailAlerts"
                  isSelected={settings.enableEmailAlerts}
                  onChange={handleChange}
                >
                  Enable Email Alerts
                </Switch>

                {settings.enableEmailAlerts && (
                  <Input
                    label="Email Address"
                    type="email"
                    name="emailAddress"
                    value={settings.emailAddress}
                    onChange={handleChange}
                  />
                )}
              </div>

              <Divider />

              <div>
                <label className="block text-sm font-medium mb-1 dark:text-gray-300">
                  Slippage Tolerance: {settings.slippageTolerance}%
                </label>
                <Slider
                  name="slippageTolerance"
                  value={settings.slippageTolerance}
                  minValue={0.1}
                  maxValue={2}
                  step={0.1}
                  onChange={(value) =>
                    handleSliderChange("slippageTolerance", value)
                  }
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Maximum allowed slippage for trade execution
                </p>
              </div>
            </div>
          )}

          {activeTab === "symbols" && (
            <div className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Select
                  label="Default Symbol"
                  name="defaultSymbol"
                  selectedKeys={[settings.defaultSymbol || symbols[0]]}
                  onSelectionChange={(keys) =>
                    handleChange({
                      target: {
                        name: "defaultSymbol",
                        value: Array.from(keys)[0],
                      },
                    })
                  }
                >
                  {symbols.map((symbol) => (
                    <SelectItem key={symbol} value={symbol}>
                      {symbol}
                    </SelectItem>
                  ))}
                </Select>

                <Input
                  label="Default Lot Size"
                  type="number"
                  name="defaultLotSize"
                  step="0.01"
                  min="0.01"
                  value={settings.defaultLotSize || 0.1}
                  onChange={handleChange}
                />
              </div>

              <Divider />

              <h3 className="text-lg font-semibold flex items-center space-x-2 dark:text-gray-100">
                <CurrencyDollarIcon className="w-5 h-5" />
                <span>Symbol-Specific Settings</span>
              </h3>

              <div className="space-y-4">
                {symbols.map((symbol) => (
                  <div
                    key={symbol}
                    className="p-4 border rounded-lg dark:border-gray-700 bg-gray-50 dark:bg-gray-700"
                  >
                    <div className="flex justify-between items-center mb-3">
                      <h4 className="font-medium dark:text-gray-100">
                        {symbol}
                      </h4>
                      <Switch
                        isSelected={symbolConfigs[symbol]?.enabled || false}
                        onChange={() =>
                          onToggleSymbol(
                            symbol,
                            !symbolConfigs[symbol]?.enabled
                          )
                        }
                        size="sm"
                      >
                        {symbolConfigs[symbol]?.enabled
                          ? "Enabled"
                          : "Disabled"}
                      </Switch>
                    </div>

                    {symbolConfigs[symbol]?.enabled && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-2">
                        <Select
                          label="Strategy"
                          selectedKeys={[
                            symbolConfigs[symbol]?.strategy || "balanced",
                          ]}
                          onSelectionChange={(keys) =>
                            onUpdateStrategy(symbol, Array.from(keys)[0])
                          }
                          size="sm"
                        >
                          <SelectItem key="aggressive">Aggressive</SelectItem>
                          <SelectItem key="balanced">Balanced</SelectItem>
                          <SelectItem key="passive">Passive</SelectItem>
                        </Select>

                        <Select
                          label="Emotion"
                          selectedKeys={[
                            symbolConfigs[symbol]?.emotion || "standard",
                          ]}
                          onSelectionChange={(keys) =>
                            onUpdateEmotion(symbol, Array.from(keys)[0])
                          }
                          size="sm"
                        >
                          <SelectItem key="greedy">Greedy</SelectItem>
                          <SelectItem key="standard">Standard</SelectItem>
                          <SelectItem key="fearful">Fearful</SelectItem>
                        </Select>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardBody>

        <div className="p-4 border-t border-gray-200 dark:border-gray-700 flex justify-end space-x-3">
          <Button
            color="danger"
            variant="light"
            onPress={handleReset}
            startContent={<ArrowPathIcon className="w-5 h-5" />}
          >
            Reset Defaults
          </Button>
          <Button
            color="primary"
            onPress={handleSave}
            isLoading={loading}
            startContent={<CogIcon className="w-5 h-5" />}
          >
            Save Settings
          </Button>
        </div>
      </Card>
    </div>
  );
};
SettingsPage.propTypes = {
  symbols: PropTypes.array.isRequired,
  symbolConfigs: PropTypes.object.isRequired,
  onToggleSymbol: PropTypes.func.isRequired,
  onUpdateEmotion: PropTypes.func.isRequired,
  onUpdateStrategy: PropTypes.func.isRequired,
  onSaveConfig: PropTypes.func.isRequired,
};

export default SettingsPage;
