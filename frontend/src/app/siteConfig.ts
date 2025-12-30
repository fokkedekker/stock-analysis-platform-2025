export const siteConfig = {
  name: "Stock Analysis",
  url: "http://localhost:3000",
  description: "Fundamental valuation systems for NYSE/NASDAQ stocks",
  baseLinks: {
    home: "/",
    portfolio: "/portfolio",
    gridSearch: "/grid-search",
    // Models section
    factorScreening: "/models/factor-screening",
    elasticNet: "/models/elastic-net",
    gam: "/models/gam",
    // Screeners
    graham: "/screeners/graham",
    magicFormula: "/screeners/magic-formula",
    piotroski: "/screeners/piotroski",
    altman: "/screeners/altman",
    roic: "/screeners/roic",
    peg: "/screeners/peg",
    famaFrench: "/screeners/fama-french",
    netNet: "/screeners/net-net",
  },
}

export type siteConfig = typeof siteConfig
