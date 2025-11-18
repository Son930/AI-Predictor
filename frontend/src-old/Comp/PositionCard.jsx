// Redesigned PositionCard.jsx
/* eslint-disable react/prop-types */
import "react";

const PositionCard = ({ position, onClose }) => {
  const {
    ticket,
    symbol,
    type,
    volume,
    profit,
    profit_pct,
    entry_price,
    current_price,
    sl,
    tp,
    should_close,
    close_reason,
  } = position;

  // Handle close with confirmation
  const handleClose = () => {
    if (window.confirm(`Are you sure you want to close position ${symbol}?`)) {
      onClose(ticket);
    }
  };

  // Format profit/loss with currency
  const formatProfit = (value) => {
    return `$${value.toFixed(2)}`; // Simple USD formatting
  };

  return (
    <div
      className={`p-4 rounded-lg shadow-md border ${
        should_close
          ? "bg-red-50 dark:bg-red-900 border-red-200 dark:border-red-700"
          : "bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700"
      } hover:shadow-lg transition-shadow`}
    >
      <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-100">
        {symbol}{" "}
        <span className="text-sm text-gray-500 dark:text-gray-400">{type}</span>
      </h3>
      <div className="mt-3 space-y-2 text-sm">
        <p className="text-gray-600 dark:text-gray-300">Volume: {volume}</p>
        <p
          className={`font-semibold flex items-center space-x-1 ${
            profit >= 0
              ? "text-green-600 dark:text-green-400"
              : "text-red-600 dark:text-red-400"
          }`}
          aria-label={`Profit: ${formatProfit(profit)}, ${profit_pct.toFixed(
            2
          )} percent`}
        >
          <span>{profit >= 0 ? "▲" : "▼"}</span>
          <span>{formatProfit(profit)}</span>
          <span className="text-gray-500 dark:text-gray-400">
            ({profit_pct.toFixed(2)}%)
          </span>
        </p>
        <p className="text-gray-600 dark:text-gray-300">
          Entry: {entry_price} | Current: {current_price}
        </p>
        <p className="text-gray-600 dark:text-gray-300">
          SL: {sl} | TP: {tp}
        </p>
      </div>
      {should_close && (
        <div className="mt-4 p-3 bg-red-100 dark:bg-red-800 text-red-700 dark:text-red-300 rounded-md text-sm">
          <p className="mb-2">{close_reason}</p>
          <button
            onClick={handleClose}
            className="w-full px-3 py-1 bg-red-500 text-white rounded-md hover:bg-red-600 transition-colors focus:outline-none focus:ring-2 focus:ring-red-400"
            aria-label={`Close position ${symbol}`}
          >
            Close Now
          </button>
        </div>
      )}
    </div>
  );
};

export default PositionCard;
