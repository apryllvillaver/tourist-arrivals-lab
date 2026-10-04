import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "Tourist Arrivals | Forecasting Lab", description: "Explore Philippine tourist arrivals, evaluate a forecast and inspect the model's reasoning." };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return <html lang="en"><body>{children}</body></html>;
}
