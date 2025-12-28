import { formatCurrency, formatNumber, formatPercent } from "./api";

// ============================================
// STAGE 1: SURVIVAL NARRATIVES
// ============================================

export function getAltmanNarrative(
  zScore: number | null | undefined,
  zone: string | null | undefined,
  symbol: string
): string {
  if (zScore == null || zone == null) {
    return `Altman Z-Score data is not available for ${symbol}.`;
  }

  const score = formatNumber(zScore, 2);

  if (zone === "safe") {
    return `The Altman Z-Score measures bankruptcy risk by analyzing working capital, retained earnings, profitability, and debt levels. ${symbol}'s score of ${score} places it firmly in the safe zone (above 3.0) - indicating strong financial health with minimal risk of bankruptcy in the next two years.`;
  } else if (zone === "grey") {
    return `The Altman Z-Score predicts bankruptcy risk using five financial ratios. ${symbol}'s score of ${score} falls in the grey zone (1.8-3.0) - not immediately concerning, but showing some financial stress signals that warrant monitoring.`;
  } else {
    return `The Altman Z-Score measures bankruptcy probability using profitability, debt, and asset ratios. At ${score}, ${symbol} falls in the distress zone (below 1.8) - historically, companies in this range have significantly elevated bankruptcy risk within two years.`;
  }
}

export function getPiotroskiNarrative(
  fScore: number | null | undefined,
  symbol: string
): string {
  if (fScore == null) {
    return `Piotroski F-Score data is not available for ${symbol}.`;
  }

  if (fScore >= 7) {
    return `The Piotroski F-Score checks 9 fundamental signals: profitability, cash flow quality, debt changes, share dilution, and operating efficiency. ${symbol} scores ${fScore}/9 - strong across the board, with fundamentals actively improving.`;
  } else if (fScore >= 5) {
    return `The Piotroski F-Score evaluates 9 fundamental health checks covering profitability, leverage, and efficiency. ${symbol}'s ${fScore}/9 shows mixed results - some areas improving while others remain flat or declining. Worth investigating which signals are failing.`;
  } else {
    return `The Piotroski F-Score awards points for positive profitability, improving margins, decreasing debt, and other fundamental improvements. ${symbol}'s low score of ${fScore}/9 suggests multiple fundamental weaknesses - several key metrics are deteriorating.`;
  }
}

export function getSurvivalSummary(
  altmanPass: boolean | null | undefined,
  piotroskiPass: boolean | null | undefined,
  symbol: string
): string {
  if (altmanPass && piotroskiPass) {
    return `${symbol} passes both survival tests. Strong balance sheet with positive fundamental momentum - this company has the financial foundation to weather tough times.`;
  } else if (altmanPass && !piotroskiPass) {
    return `Balance sheet looks solid, but fundamentals show some weakness. The company isn't at risk of failing, but operational improvements would strengthen the picture.`;
  } else if (!altmanPass && piotroskiPass) {
    return `Fundamentals are trending positive, but the balance sheet carries some risk. Watch debt levels and liquidity carefully.`;
  } else {
    return `Warning flags on both fronts. Elevated bankruptcy risk combined with weak fundamentals - requires extra caution and deeper due diligence.`;
  }
}

// ============================================
// STAGE 2: QUALITY NARRATIVES
// ============================================

export function getROICNarrative(
  roic: number | null | undefined,
  symbol: string
): string {
  if (roic == null) {
    return `Return on Invested Capital (ROIC) data is not available for ${symbol}.`;
  }

  const roicPercent = (roic * 100).toFixed(1);
  const roicDollars = (roic * 100).toFixed(0);

  if (roic >= 0.20) {
    return `Return on Invested Capital (ROIC) measures how efficiently a company converts capital into profit. ${symbol}'s ${roicPercent}% ROIC means for every $100 invested in the business, it generates $${roicDollars} in operating profit. This exceptional return suggests strong competitive advantages - a moat that lets the company earn outsized returns.`;
  } else if (roic >= 0.15) {
    return `ROIC measures profit efficiency - what the company earns on money invested in operations. At ${roicPercent}%, ${symbol} earns solid returns well above its cost of capital (typically 8-10%). This indicates a quality business, though not quite in the elite 'compounder' category.`;
  } else if (roic >= 0.08) {
    return `ROIC shows how much profit a company generates per dollar invested. ${symbol}'s ${roicPercent}% roughly covers its cost of capital (what investors expect to earn). The business creates modest value but lacks the exceptional returns that signal competitive advantages.`;
  } else {
    return `Return on Invested Capital measures profit efficiency. At ${roicPercent}%, ${symbol} earns less than its cost of capital (typically 8-10%) - meaning the company may destroy value over time. Each dollar invested generates subpar returns compared to alternatives.`;
  }
}

export function getFCFNarrative(
  fcf: number | null | undefined,
  fcfPositive5yr: boolean | null | undefined,
  symbol: string
): string {
  if (fcf == null) {
    return `Free Cash Flow data is not available for ${symbol}.`;
  }

  const fcfFormatted = formatCurrency(fcf);

  if (fcf > 0 && fcfPositive5yr) {
    return `Free Cash Flow represents real cash generated after all expenses and investments. ${symbol} generated ${fcfFormatted} in FCF with 5+ consecutive positive years - a reliable cash generator. Consistent FCF is crucial because it proves the business actually produces cash, not just accounting profits.`;
  } else if (fcf > 0) {
    return `Free Cash Flow is the real cash remaining after operations and investments. ${symbol}'s current FCF of ${fcfFormatted} is positive, but hasn't been consistent every year - some years saw negative cash flow. Investigate what causes the fluctuations.`;
  } else {
    return `Free Cash Flow measures actual cash generated after all expenses. ${symbol}'s negative FCF of ${fcfFormatted} means the business consumes more cash than it produces - either through heavy investment or operational issues. This requires external funding to sustain.`;
  }
}

export function getQualityTagNarrative(tag: string): string {
  const narratives: Record<string, string> = {
    "Durable Compounder":
      "This company shows stable ROIC and gross margins over 5+ years - the hallmark of a durable competitive advantage (or 'moat'). Consistent returns suggest the business can maintain its edge over competitors.",
    "Cash Machine":
      "Free cash flow exceeds reported net income - a sign of high-quality earnings. Some companies report profits that never convert to cash; this one generates more cash than it reports in earnings.",
    "Deep Value":
      "Valuation metrics suggest the market is overly pessimistic. High FCF yield or low EV/EBIT can indicate the stock is priced for failure when fundamentals don't justify such pessimism.",
    "Heavy Reinvestor":
      "This company plows significant cash back into R&D and capital expenditures. Heavy reinvestment can fuel future growth but means less cash returned to shareholders today.",
    "Premium Priced":
      "Low FCF yield and/or high EV/EBIT suggest the market has high expectations built into the price. Future performance must be exceptional just to justify current valuation.",
    "Volatile Returns":
      "ROIC swings significantly year to year - the business may be cyclical, face intense competition, or have unpredictable economics. Harder to value and predict.",
    "Weak Moat Signal":
      "Gross margins have fluctuated over time, which can indicate competitive pressure or lack of pricing power. Stable margins typically signal a defensible market position.",
    "Earnings Quality Concern":
      "Cash flow significantly trails reported earnings - a potential red flag. When companies report profits but don't generate corresponding cash, investigate where the 'earnings' are going.",
  };

  return narratives[tag] || `${tag} - No description available.`;
}

export function getQualitySummary(
  qualityLabel: string,
  symbol: string
): string {
  if (qualityLabel === "Compounder") {
    return `${symbol} qualifies as a "Compounder" - high returns on capital with consistent cash generation. These are the businesses that can compound wealth over long periods.`;
  } else if (qualityLabel === "Average") {
    return `${symbol} shows average quality metrics - decent but not exceptional. The business covers its cost of capital without demonstrating obvious competitive advantages.`;
  } else {
    return `${symbol} shows weak quality metrics. Low returns on capital and/or inconsistent cash flow raise questions about the business model's viability.`;
  }
}

// ============================================
// STAGE 3: VALUATION NARRATIVES
// ============================================

export function getGrahamNarrative(
  criteriaPassed: number | null | undefined,
  symbol: string
): string {
  if (criteriaPassed == null) {
    return `Graham analysis data is not available for ${symbol}.`;
  }

  if (criteriaPassed >= 6) {
    return `Benjamin Graham's value investing checklist examines financial strength, earnings consistency, dividends, and valuation. ${symbol} passes ${criteriaPassed}/8 criteria - qualifying as a classic 'Graham stock' with multiple safety factors aligned. Graham believed such stocks offered a margin of safety against permanent capital loss.`;
  } else if (criteriaPassed >= 4) {
    return `Graham's checklist tests for financially strong, consistently profitable companies trading at reasonable valuations. ${symbol} passes ${criteriaPassed}/8 criteria - showing value characteristics but falling short of the full Graham standard. Some safety factors are missing.`;
  } else {
    return `Benjamin Graham's 8-point checklist identifies conservatively valued, financially sound companies. ${symbol} passes only ${criteriaPassed}/8 - this isn't a traditional Graham value stock. Either too expensive, too small, or lacking the consistent earnings Graham demanded.`;
  }
}

export function getNetNetNarrative(
  discountToNcav: number | null | undefined,
  tradingBelowNcav: boolean | null | undefined,
  symbol: string
): string {
  if (discountToNcav == null) {
    return `Net-Net (NCAV) analysis data is not available for ${symbol}.`;
  }

  const discountPercent = formatPercent(discountToNcav);

  if (tradingBelowNcav) {
    return `Net Current Asset Value (NCAV) = current assets minus ALL liabilities. It's the theoretical liquidation value - what shareholders would get if the company closed shop today. ${symbol} trades at ${discountPercent} of NCAV, meaning you're buying assets at a discount. Even in liquidation, you'd theoretically profit.`;
  } else {
    return `Net-Net investing means buying below liquidation value (NCAV = current assets - total liabilities). ${symbol} trades at ${discountPercent} of NCAV - well above liquidation value. The market values the ongoing business, not just the assets. This is normal; true net-nets are extremely rare.`;
  }
}

export function getPEGNarrative(
  pegRatio: number | null | undefined,
  _pegPass: boolean | null | undefined,
  symbol: string
): string {
  if (pegRatio == null) {
    return `PEG ratio data is not available for ${symbol}. This could mean growth rate data is unavailable or the company isn't profitable.`;
  }

  const peg = formatNumber(pegRatio, 2);

  if (pegRatio < 1.0) {
    return `The PEG ratio divides P/E by earnings growth rate. ${symbol}'s PEG of ${peg} means you're paying less than 1x for each percent of expected growth - potentially undervalued if growth materializes. A P/E of 20 with 25% growth would give PEG of 0.8.`;
  } else if (pegRatio <= 1.5) {
    return `PEG adjusts P/E for growth rate - a stock with P/E 25 and 25% growth has PEG of 1.0. ${symbol}'s PEG of ${peg} suggests fair value - you're paying proportionally for the growth expected. Not a screaming bargain, but not overpriced either.`;
  } else {
    return `The PEG ratio (P/E divided by growth rate) helps assess if you're overpaying for growth. ${symbol}'s PEG of ${peg} indicates premium pricing - you're paying more than 1.5x for each percent of growth. High expectations are baked in; growth must exceed forecasts to justify the price.`;
  }
}

export function getMagicFormulaNarrative(
  combinedRank: number | null | undefined,
  symbol: string
): string {
  if (combinedRank == null) {
    return `Magic Formula ranking data is not available for ${symbol}.`;
  }

  if (combinedRank <= 100) {
    return `The Magic Formula ranks stocks by combining quality (Return on Capital) and cheapness (Earnings Yield). ${symbol} ranks #${combinedRank} - in the top tier combining high-quality operations with attractive valuation. Greenblatt's backtests showed portfolios of top-ranked stocks significantly outperformed.`;
  } else if (combinedRank <= 500) {
    return `Greenblatt's Magic Formula ranks companies by quality + cheapness. ${symbol}'s rank of #${combinedRank} is respectable but not elite - either good quality at fair prices, or decent quality at good prices, but not the best of both worlds.`;
  } else {
    return `The Magic Formula seeks stocks that are both high-quality AND cheap. ${symbol}'s rank of #${combinedRank} means it doesn't score well on this combination - either too expensive for its quality, or too low quality for its price, or both.`;
  }
}

export function getFamaFrenchNarrative(
  bookToMarketPercentile: number | null | undefined,
  symbol: string
): string {
  if (bookToMarketPercentile == null) {
    return `Fama-French factor data is not available for ${symbol}.`;
  }

  const pct = Math.round(bookToMarketPercentile * 100);

  if (bookToMarketPercentile >= 0.7) {
    return `Book-to-Market ratio compares accounting value to market value. Academic research shows high B/M stocks ('value') historically outperform low B/M ('growth') by 3-5% annually. ${symbol} at the ${pct}th percentile sits firmly in value territory - statistically positioned for higher expected returns.`;
  } else if (bookToMarketPercentile >= 0.3) {
    return `The Fama-French value factor uses Book-to-Market ratio to classify stocks. Research shows high B/M stocks outperform long-term. ${symbol} at ${pct}th percentile falls in the middle - neither deep value nor high growth. Neutral factor exposure.`;
  } else {
    return `Book-to-Market ratio is used to identify value vs growth stocks. High B/M stocks historically outperform. ${symbol} at ${pct}th percentile is classified as a 'growth' stock - market values it well above book value. Growth stocks can win but don't have the historical value premium tailwind.`;
  }
}

export function getValuationSummary(
  lensesPassedCount: number,
  lensesPassed: string[]
): string {
  if (lensesPassedCount === 0) {
    return `No valuation lens sees this stock as undervalued. The price already reflects high expectations or the business quality doesn't justify a value label.`;
  } else if (lensesPassedCount === 1) {
    return `One valuation lens (${lensesPassed.join(", ")}) sees potential value here. Not a strong consensus, but worth investigating further.`;
  } else if (lensesPassedCount >= 3) {
    return `Multiple valuation lenses agree (${lensesPassed.join(", ")}) - a rare convergence suggesting genuine undervaluation. When different methodologies point the same direction, conviction increases.`;
  } else {
    return `Two valuation lenses (${lensesPassed.join(", ")}) identify value. Some agreement across methodologies, though not a strong consensus.`;
  }
}

// ============================================
// STAGE 4: FACTOR EXPOSURE NARRATIVES
// ============================================

export function getSizeNarrative(
  marketCap: number | null | undefined,
  symbol: string
): string {
  if (marketCap == null) {
    return `Market cap data is not available for ${symbol}.`;
  }

  const mcapFormatted = formatCurrency(marketCap);

  if (marketCap >= 10e9) {
    return `At ${mcapFormatted} market cap, ${symbol} is a large-cap stock. Large-caps tend to be more stable with greater analyst coverage and liquidity. The tradeoff: smaller companies historically deliver higher returns (the 'size premium') over long periods.`;
  } else if (marketCap >= 2e9) {
    return `${symbol}'s ${mcapFormatted} market cap puts it in mid-cap territory - a balance between growth potential and stability. Mid-caps can offer both reasonable liquidity and room to grow.`;
  } else {
    return `At ${mcapFormatted}, ${symbol} is a small-cap. Small companies have historically outperformed large ones, but with higher volatility and less liquidity. Expect bigger swings and potentially thinner trading volume.`;
  }
}

export function getProfitabilityNarrative(
  profitabilityPercentile: number | null | undefined,
  symbol: string
): string {
  if (profitabilityPercentile == null) {
    return `Profitability factor data is not available for ${symbol}.`;
  }

  const pct = Math.round(profitabilityPercentile * 100);

  if (profitabilityPercentile >= 0.7) {
    return `${symbol}'s profitability ranks in the ${pct}th percentile - among the most profitable companies relative to assets. Research shows high-profitability stocks tend to outperform.`;
  } else if (profitabilityPercentile >= 0.3) {
    return `Profitability at ${pct}th percentile - average relative to peers. Neither a profitability premium nor penalty expected.`;
  } else {
    return `Profitability at ${pct}th percentile places ${symbol} among the less profitable companies. Lower profitability is associated with lower expected returns in academic research.`;
  }
}

export function getFactorSummary(
  marketCap: number | null | undefined,
  bookToMarketPercentile: number | null | undefined,
  profitabilityPercentile: number | null | undefined,
  symbol: string
): string {
  // Determine size
  let size = "unknown-sized";
  if (marketCap != null) {
    size = marketCap >= 10e9 ? "large-cap" : marketCap >= 2e9 ? "mid-cap" : "small-cap";
  }

  // Determine value/growth
  let valueGrowth = "neutral";
  if (bookToMarketPercentile != null) {
    valueGrowth = bookToMarketPercentile >= 0.7 ? "value" : bookToMarketPercentile <= 0.3 ? "growth" : "blend";
  }

  // Determine profitability
  let profitability = "average";
  if (profitabilityPercentile != null) {
    profitability = profitabilityPercentile >= 0.7 ? "high" : profitabilityPercentile <= 0.3 ? "low" : "medium";
  }

  // Generate implication
  let implication = "";
  if (valueGrowth === "value" && profitability === "high") {
    implication = "historically favorable factor tilts - value and profitability premiums both support expected returns.";
  } else if (valueGrowth === "growth" && profitability === "low") {
    implication = "factor headwinds - neither value nor profitability premiums on your side.";
  } else if (valueGrowth === "value") {
    implication = "the value premium may support returns, though profitability factor is neutral or negative.";
  } else if (profitability === "high") {
    implication = "the profitability premium may support returns, though not positioned for value premium.";
  } else {
    implication = "neutral factor positioning - no strong tailwinds or headwinds from academic factors.";
  }

  return `${symbol} is a ${size} ${valueGrowth} stock with ${profitability} profitability. Factor profile suggests ${implication}`;
}
