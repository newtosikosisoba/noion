import { create } from "zustand";
import { authApi, type UserData } from "./api";

interface AuthState {
  user: UserData | null;
  loading: boolean;
  fetchUser: () => Promise<void>;
  setUser: (u: UserData | null) => void;
  logout: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,
  fetchUser: async () => {
    try {
      const res = await authApi.me();
      set({ user: res.data, loading: false });
    } catch {
      set({ user: null, loading: false });
    }
  },
  setUser: (user) => set({ user, loading: false }),
  logout: async () => {
    await authApi.logout();
    set({ user: null });
  },
}));
