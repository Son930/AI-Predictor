import BotDashboard from "../Comp/BotDashboard";
import "../../src/App.css";

function Home() {
  return (
    <div className="relative min-h-screen">
      {/* Background Grid Layer */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#4f4f4f2e_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:14px_24px]"></div>

      {/* Radial Gradient Layer */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_40vw_at_50%_30%,#fbfbfb36,#000)]"></div>

      {/* Content Layer */}
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-12">
          <h1 className="text-4xl sm:text-6xl md:text-8xl lg:text-9xl font-bold text-white font-[Superline]">
            Trading <span className="font-[SuperlineOutline]">Bot</span>{" "}
            Dashboard
          </h1>
          <h2 className="text-lg sm:text-xl text-white font-[Clofie] mt-4">
            Developed By Kicker93 - Sunny Sondhi
          </h2>
        </div>
        <BotDashboard />
      </div>
    </div>
  );
}

export default Home;
