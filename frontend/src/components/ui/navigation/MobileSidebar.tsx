import { siteConfig } from "@/app/siteConfig"
import { Button } from "@/components/Button"
import {
  Drawer,
  DrawerBody,
  DrawerClose,
  DrawerContent,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/Drawer"
import { StockSearch } from "@/components/StockSearch"
import { cx, focusRing } from "@/lib/utils"
import {
  RiBarChartBoxLine,
  RiFlaskLine,
  RiHome2Line,
  RiLineChartLine,
  RiMenuLine,
  RiPercentLine,
  RiScales3Line,
  RiShieldCheckLine,
  RiSparklingLine,
  RiStackLine,
  RiStockLine,
} from "@remixicon/react"
import Link from "next/link"
import { usePathname } from "next/navigation"

const navigation = [
  { name: "Pipeline", href: siteConfig.baseLinks.home, icon: RiHome2Line },
  { name: "Grid Search", href: siteConfig.baseLinks.gridSearch, icon: RiFlaskLine },
] as const

const screeners = [
  { name: "Graham", href: siteConfig.baseLinks.graham, icon: RiScales3Line },
  { name: "Magic Formula", href: siteConfig.baseLinks.magicFormula, icon: RiSparklingLine },
  { name: "Piotroski", href: siteConfig.baseLinks.piotroski, icon: RiBarChartBoxLine },
  { name: "Altman Z-Score", href: siteConfig.baseLinks.altman, icon: RiShieldCheckLine },
  { name: "ROIC", href: siteConfig.baseLinks.roic, icon: RiPercentLine },
  { name: "PEG", href: siteConfig.baseLinks.peg, icon: RiLineChartLine },
  { name: "Fama-French", href: siteConfig.baseLinks.famaFrench, icon: RiStackLine },
  { name: "Net-Net", href: siteConfig.baseLinks.netNet, icon: RiStockLine },
] as const

export default function MobileSidebar() {
  const pathname = usePathname()
  const isActive = (itemHref: string) => {
    if (itemHref === "/") {
      return pathname === "/"
    }
    return pathname === itemHref || pathname.startsWith(itemHref)
  }
  return (
    <>
      <Drawer>
        <DrawerTrigger asChild>
          <Button
            variant="ghost"
            aria-label="open sidebar"
            className="group flex items-center rounded-md p-2 text-sm font-medium hover:bg-gray-100 data-[state=open]:bg-gray-100 data-[state=open]:bg-gray-400/10 hover:dark:bg-gray-400/10"
          >
            <RiMenuLine
              className="size-6 shrink-0 sm:size-5"
              aria-hidden="true"
            />
          </Button>
        </DrawerTrigger>
        <DrawerContent className="sm:max-w-lg">
          <DrawerHeader>
            <DrawerTitle>Stock Analysis</DrawerTitle>
          </DrawerHeader>
          <DrawerBody>
            <StockSearch />
            <nav
              aria-label="core mobile navigation links"
              className="flex flex-1 flex-col space-y-10 mt-4"
            >
              <ul role="list" className="space-y-1.5">
                {navigation.map((item) => (
                  <li key={item.name}>
                    <DrawerClose asChild>
                      <Link
                        href={item.href}
                        className={cx(
                          isActive(item.href)
                            ? "text-indigo-600 dark:text-indigo-400"
                            : "text-gray-600 hover:text-gray-900 dark:text-gray-400 hover:dark:text-gray-50",
                          "flex items-center gap-x-2.5 rounded-md px-2 py-1.5 text-base font-medium transition hover:bg-gray-100 sm:text-sm hover:dark:bg-gray-900",
                          focusRing,
                        )}
                      >
                        <item.icon
                          className="size-5 shrink-0"
                          aria-hidden="true"
                        />
                        {item.name}
                      </Link>
                    </DrawerClose>
                  </li>
                ))}
              </ul>
              <div>
                <span className="text-sm font-medium leading-6 text-gray-500 sm:text-xs">
                  Screeners
                </span>
                <ul aria-label="screeners" role="list" className="space-y-0.5">
                  {screeners.map((item) => (
                    <li key={item.name}>
                      <DrawerClose asChild>
                        <Link
                          href={item.href}
                          className={cx(
                            isActive(item.href)
                              ? "text-indigo-600 dark:text-indigo-400"
                              : "text-gray-700 hover:text-gray-900 dark:text-gray-400 hover:dark:text-gray-50",
                            "flex items-center gap-x-2.5 rounded-md px-2 py-1.5 font-medium transition hover:bg-gray-100 sm:text-sm hover:dark:bg-gray-900",
                            focusRing,
                          )}
                        >
                          <item.icon
                            className="size-4 shrink-0"
                            aria-hidden="true"
                          />
                          {item.name}
                        </Link>
                      </DrawerClose>
                    </li>
                  ))}
                </ul>
              </div>
            </nav>
          </DrawerBody>
        </DrawerContent>
      </Drawer>
    </>
  )
}
