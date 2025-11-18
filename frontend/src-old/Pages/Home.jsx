// Redesigned Home.jsx
import BotDashboard from "../Comp/BotDashboard";

function Home() {
  return (
    <div className="relative min-h-screen bg-gray-50 dark:bg-gray-900">
      {/* Background Grid Layer */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#e5e7eb_1px,transparent_1px),linear-gradient(to_bottom,#e5e7eb_1px,transparent_1px)] dark:bg-[linear-gradient(to_right,#374151_1px,transparent_1px),linear-gradient(to_bottom,#374151_1px,transparent_1px)] bg-[size:14px_24px]"></div>

      {/* Radial Gradient Layer */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_40vw_at_50%_30%,#f3f4f636,#111827_dark:#000000)] dark:bg-[radial-gradient(circle_40vw_at_50%_30%,#1f293736,#000000)]"></div>

      {/* Content Layer */}
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12 pt-8">
          <h1 className="text-4xl sm:text-6xl md:text-8xl lg:text-9xl font-bold text-gray-900 dark:text-white font-[Superline]">
            Trading <span className="font-[SuperlineOutline]">Bot</span>{" "}
            Dashboard
          </h1>
          <h2 className="text-lg sm:text-xl text-gray-600 dark:text-gray-300 font-[Clofie] mt-4">
            Developed By Kicker93 - Sunny Sondhi
          </h2>
        </div>
        <BotDashboard />
      </div>
    </div>
  );
}

export default Home;
