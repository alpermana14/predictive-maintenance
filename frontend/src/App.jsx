// frontend/src/App.jsx
import Dashboard from "./components/Dashboard";
import Chatbot from "./components/Chatbot"; // <--- Import this

function App() {
  return (
    <div>
      <Dashboard />
      <Chatbot />  {/* <--- Add this */}
    </div>
  );
}
export default App;