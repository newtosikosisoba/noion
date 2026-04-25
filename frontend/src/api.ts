import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  withCredentials: true,
});

export interface UserData {
  id: string;
  email: string;
  display_name: string | null;
  tier: string;
  created_at: string;
}

export interface JobOutputs {
  midi_full: string | null;
  midi_piano: string | null;
  midi_bass: string | null;
  midi_drums: string | null;
  wav: string | null;
}

export interface JobData {
  id: string;
  status: string;
  progress: number;
  progress_message: string | null;
  original_filename: string;
  tier_at_creation: string;
  outputs: JobOutputs | null;
  created_at: string;
  completed_at: string | null;
}

export const authApi = {
  register: (email: string, password: string, display_name?: string) =>
    api.post<UserData>("/auth/register", { email, password, display_name }),
  login: (email: string, password: string) =>
    api.post<UserData>("/auth/login", { email, password }),
  logout: () => api.post("/auth/logout"),
  me: () => api.get<UserData>("/auth/me"),
};

export const jobsApi = {
  create: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return api.post<{ job_id: string }>("/jobs", form);
  },
  get: (id: string) => api.get<JobData>(`/jobs/${id}`),
  list: (skip = 0, limit = 20) =>
    api.get<{ jobs: JobData[]; total: number }>("/jobs", { params: { skip, limit } }),
};

export const billingApi = {
  checkout: () => api.post<{ url: string }>("/billing/checkout"),
  portal: () => api.post<{ url: string }>("/billing/portal"),
};

export default api;
