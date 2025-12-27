"use client"
import { siteConfig } from "@/app/siteConfig"
import { QuarterSelector } from "@/components/QuarterSelector"
import { StockSearch } from "@/components/StockSearch"
import { cx, focusRing } from "@/lib/utils"
import {
  RiBarChartBoxLine,
  RiFilter3Line,
  RiLineChartLine,
  RiMoonLine,
  RiPercentLine,
  RiScales3Line,
  RiShieldCheckLine,
  RiSparklingLine,
  RiStackLine,
  RiStockLine,
  RiSunLine,
} from "@remixicon/react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useTheme } from "next-themes"
import { useEffect, useState } from "react"
import MobileSidebar from "./MobileSidebar"

const navigation = [
  { name: "Pipeline", href: siteConfig.baseLinks.home, icon: RiFilter3Line },
] as const

// Grouped by pipeline stage
const survivalGates = [
  { name: "Altman Z-Score", href: siteConfig.baseLinks.altman, icon: RiShieldCheckLine },
  { name: "Piotroski", href: siteConfig.baseLinks.piotroski, icon: RiBarChartBoxLine },
] as const

const qualityScreens = [
  { name: "ROIC", href: siteConfig.baseLinks.roic, icon: RiPercentLine },
  { name: "Fama-French", href: siteConfig.baseLinks.famaFrench, icon: RiStackLine },
] as const

const valuationLenses = [
  { name: "Graham", href: siteConfig.baseLinks.graham, icon: RiScales3Line },
  { name: "Magic Formula", href: siteConfig.baseLinks.magicFormula, icon: RiSparklingLine },
  { name: "PEG", href: siteConfig.baseLinks.peg, icon: RiLineChartLine },
  { name: "Net-Net", href: siteConfig.baseLinks.netNet, icon: RiStockLine },
] as const

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = useState(false)

  // Avoid hydration mismatch
  useEffect(() => setMounted(true), [])

  if (!mounted) {
    return (
      <div className="flex items-center gap-2 px-2 py-1.5">
        <div className="w-4 h-4" />
        <span className="text-sm text-gray-500">Theme</span>
      </div>
    )
  }

  return (
    <button
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      className={cx(
        "flex items-center gap-x-2.5 rounded-md px-2 py-1.5 text-sm font-medium w-full",
        "text-gray-700 hover:text-gray-900 dark:text-gray-400 hover:dark:text-gray-50",
        "transition hover:bg-gray-100 hover:dark:bg-gray-900",
        focusRing
      )}
    >
      {theme === "dark" ? (
        <>
          <RiSunLine className="size-4 shrink-0" aria-hidden="true" />
          Light Mode
        </>
      ) : (
        <>
          <RiMoonLine className="size-4 shrink-0" aria-hidden="true" />
          Dark Mode
        </>
      )}
    </button>
  )
}

function NavSection({
  title,
  items,
  isActive,
}: {
  title: string
  items: readonly { name: string; href: string; icon: React.ComponentType<{ className?: string }> }[]
  isActive: (href: string) => boolean
}) {
  return (
    <div>
      <span className="text-xs font-medium leading-6 text-gray-500">
        {title}
      </span>
      <ul aria-label={title} role="list" className="space-y-0.5">
        {items.map((item) => (
          <li key={item.name}>
            <Link
              href={item.href}
              className={cx(
                isActive(item.href)
                  ? "text-indigo-600 dark:text-indigo-400"
                  : "text-gray-700 hover:text-gray-900 dark:text-gray-400 hover:dark:text-gray-50",
                "flex items-center gap-x-2.5 rounded-md px-2 py-1.5 text-sm font-medium transition hover:bg-gray-100 hover:dark:bg-gray-900",
                focusRing,
              )}
            >
              <item.icon className="size-4 shrink-0" aria-hidden="true" />
              {item.name}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function Sidebar() {
  const pathname = usePathname()
  const isActive = (itemHref: string) => {
    if (itemHref === "/") {
      return pathname === "/"
    }
    return pathname === itemHref || pathname.startsWith(itemHref)
  }
  return (
    <>
      {/* sidebar (lg+) */}
      <nav className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
        <aside className="flex grow flex-col gap-y-6 overflow-y-auto border-r border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-950">
          <div className="flex items-center gap-3 px-2">
            <RiStockLine className="size-6 text-indigo-600" />
            <span className="text-lg font-semibold text-gray-900 dark:text-gray-50">
              Stock Analysis
            </span>
          </div>
          <StockSearch />
          <QuarterSelector />
          <nav
            aria-label="core navigation links"
            className="flex flex-1 flex-col space-y-6"
          >
            {/* Main navigation */}
            <ul role="list" className="space-y-0.5">
              {navigation.map((item) => (
                <li key={item.name}>
                  <Link
                    href={item.href}
                    className={cx(
                      isActive(item.href)
                        ? "text-indigo-600 dark:text-indigo-400 bg-indigo-50 dark:bg-indigo-950"
                        : "text-gray-700 hover:text-gray-900 dark:text-gray-400 hover:dark:text-gray-50",
                      "flex items-center gap-x-2.5 rounded-md px-2 py-1.5 text-sm font-medium transition hover:bg-gray-100 hover:dark:bg-gray-900",
                      focusRing,
                    )}
                  >
                    <item.icon className="size-4 shrink-0" aria-hidden="true" />
                    {item.name}
                  </Link>
                </li>
              ))}
            </ul>

            {/* Grouped screeners */}
            <NavSection title="Survival Gates" items={survivalGates} isActive={isActive} />
            <NavSection title="Quality Screens" items={qualityScreens} isActive={isActive} />
            <NavSection title="Valuation Lenses" items={valuationLenses} isActive={isActive} />
          </nav>
          <div className="mt-auto space-y-3">
            <ThemeToggle />
            <div className="px-2 text-xs text-gray-500">
              Connected to API: localhost:8000
            </div>
          </div>
        </aside>
      </nav>
      {/* top navbar (xs-lg) */}
      <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center justify-between border-b border-gray-200 bg-white px-2 shadow-sm sm:gap-x-6 sm:px-4 lg:hidden dark:border-gray-800 dark:bg-gray-950">
        <div className="flex items-center gap-2">
          <RiStockLine className="size-5 text-indigo-600" />
          <span className="font-semibold text-gray-900 dark:text-gray-50">
            Stock Analysis
          </span>
        </div>
        <MobileSidebar />
      </div>
    </>
  )
}
