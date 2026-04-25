import { useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import { useAuthStore } from "./store";
import Layout from "./components/Layout";
import HomePage from "./pages/HomePage";
import LoginPage from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import DashboardPage from "./pages/DashboardPage";
import JobPage from "./pages/JobPage";
import PricingPage from "./pages/PricingPage";

export default function App() {
  const fetchUser = useAuthStore((s) => s.fetchUser);

  useEffect(() => {
    fetchUser();
  }, [fetchUser]);

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/jobs/:id" element={<JobPage />} />
        <Route path="/pricing" element={<PricingPage />} />
      </Routes>
    </Layout>
  );
}
