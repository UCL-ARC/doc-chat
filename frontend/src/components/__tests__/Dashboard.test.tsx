import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { BrowserRouter } from "react-router-dom";
import axios from "axios";
import Dashboard from "../Dashboard";
import { AuthProvider } from "../../contexts/AuthContext";

jest.mock("axios");
const mockedAxios = axios as jest.Mocked<typeof axios>;

const mockNavigate = jest.fn();

jest.mock("react-router-dom", () => ({
  ...jest.requireActual("react-router-dom"),
  useNavigate: () => mockNavigate,
}));

describe("Dashboard Component", () => {
  const mockDocuments = [
    {
      id: 1,
      filename: "test.pdf",
      file_type: "application/pdf",
      created_at: "2024-03-21T00:00:00.000Z",
    },
  ];

  beforeEach(() => {
    mockedAxios.get.mockResolvedValue({ data: mockDocuments });
    render(
      <BrowserRouter>
        <AuthProvider>
          <Dashboard />
        </AuthProvider>
      </BrowserRouter>,
    );
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders dashboard with document list", async () => {
    expect(screen.getByText(/document analysis/i)).toBeInTheDocument();
    expect(screen.getByText(/drag and drop files here/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("test.pdf")).toBeInTheDocument();
    });
  });

  it("shows error message when document fetch fails", async () => {
    mockedAxios.get.mockRejectedValueOnce(new Error("Failed to fetch"));
    render(
      <BrowserRouter>
        <AuthProvider>
          <Dashboard />
        </AuthProvider>
      </BrowserRouter>,
    );

    await waitFor(() => {
      expect(
        screen.getByText(/failed to fetch documents/i),
      ).toBeInTheDocument();
    });
  });

  it("handles logout", () => {
    const logoutButton = screen.getByRole("button", { name: /logout/i });
    fireEvent.click(logoutButton);
    expect(mockNavigate).toHaveBeenCalledWith("/login");
  });
});
