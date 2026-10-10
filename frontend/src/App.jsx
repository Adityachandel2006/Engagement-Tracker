import { useState, useEffect } from "react"
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Cell
} from "recharts"

const API = "http://localhost:8000"
const COLORS = { positive: "#4ade80", negative: "#f87171", neutral: "#94a3b8" }

const COUNTRIES = [
  { id: "us", name: "United States", short: "US" },
  { id: "gb", name: "United Kingdom", short: "UK" },
  { id: "in", name: "India", short: "IN" },
  { id: "ca", name: "Canada", short: "CA" },
  { id: "au", name: "Australia", short: "AU" },
]

export default function App() {
  const [query, setQuery] = useState("")
  const [category, setCategory] = useState("")
  const [data, setData] = useState(null)
  const [compareData, setCompareData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [history, setHistory] = useState([])
  const [selectedCountries, setSelectedCountries] = useState([])

    useEffect(() => { fetch(`${API}/topics`).then((response) => response.json()).then((json) => setHistory(json.topics || [])).catch(() => {}) }, [])

  const toggleCountry = (id) => {
    if (selectedCountries.includes(id)) {
      setSelectedCountries(selectedCountries.filter(c => c !== id));
    } else {
      if (selectedCountries.length >= 3) return;
      setSelectedCountries([...selectedCountries, id]);
    }
  }

  const search = async (topic) => {
    const q = (topic || query).trim()
    if (!q) return
    setLoading(true)
    setError("")
    setData(null)
    setCompareData(null)
    try {
      let url = `${API}/search?topic=${encodeURIComponent(q)}`
      if (category) url += `&category=${encodeURIComponent(category)}`
      const res = await fetch(url)
      if (!res.ok) throw new Error((await res.json()).detail || "Not found")
      const json = await res.json()
      setData(json)
      setHistory(prev => [...new Set([json.topic, ...prev])].slice(0, 8))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const compare = async () => {
    const q = query.trim()
    if (!q || selectedCountries.length === 0) return
    setLoading(true)
    setError("")
    setData(null)
    setCompareData(null)
    try {
      let url = `${API}/compare?topic=${encodeURIComponent(q)}&countries=${selectedCountries.join(",")}`
      if (category) url += `&category=${encodeURIComponent(category)}`
      const res = await fetch(url)
      if (!res.ok) throw new Error((await res.json()).detail || "Error fetching comparison")
      const json = await res.json()
      setCompareData(json)
      setHistory(prev => [...new Set([json.topic, ...prev])].slice(0, 8))
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const sentimentBar = data
    ? [
        { name: "Positive", value: data.sentiment_breakdown.positive, color: COLORS.positive },
        { name: "Negative", value: data.sentiment_breakdown.negative, color: COLORS.negative },
        { name: "Neutral",  value: data.sentiment_breakdown.neutral,  color: COLORS.neutral },
      ]
    : []

    const chartTooltip = {
      contentStyle: { background: "#17211f", border: "0", borderRadius: 8, color: "#fff" },
      labelStyle: { color: "#b7c5bf" },
    }

    function SentimentLegend({ breakdown }) {
      return (
        <div className="sentiment-legend">
          {Object.entries(breakdown).map(([name, value]) => (
            <div className="legend-item" key={name}>
              <span className="legend-dot" style={{ background: COLORS[name] }} />
              <span>{name}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
      )
    }

    function SentimentBar({ percentages }) {
      return (
        <div className="sentiment-bar-wrap">
          <div className="sentiment-bar" aria-label="Sentiment percentage breakdown">
            {Object.entries(percentages).map(([name, value]) => <span key={name} className={`sentiment-segment ${name}`} style={{ width: `${value}%` }} title={`${name}: ${value}%`} />)}
          </div>
          <div className="percentage-labels">{Object.entries(percentages).map(([name, value]) => <span key={name}><i className={`legend-dot ${name}`} />{name} <b>{value}%</b></span>)}</div>
        </div>
      )
    }

    function ChartHeader({ eyebrow, title, detail }) {
      return (
        <div className="section-heading">
          <div>
            <span className="eyebrow">{eyebrow}</span>
            <h2>{title}</h2>
          </div>
          {detail && <span className="muted-note">{detail}</span>}
        </div>
      )
    }

    function HeadlineList({ articles, compact = false }) {
      return (
        <div className={`headline-list${compact ? " compact" : ""}`}>
          {articles?.length ? articles.map((article, index) => (
            <article className="headline" key={`${article.text}-${index}`}>
              <span className="headline-marker" style={{ background: COLORS[article.sentiment] || COLORS.neutral }} />
              <div>
                <div className="headline-meta">{article.date || "Recent"} <span>/</span> {article.sentiment || "neutral"}</div>
                <p>{article.text}</p>
              </div>
            </article>
          )) : <p className="empty-copy">No headlines available for this view.</p>}
        </div>
      )
    }

    function SignalPanel({ signals }) {
      return (
        <section className="signals-panel">
          <div className="signals-heading"><div><span className="eyebrow">Decision support</span><h2>What changed</h2></div><span className="muted-note">Signals from this report</span></div>
          <div className="signals-grid">
            {signals.map((signal) => (
              <div className={`signal-card ${signal.tone}`} key={signal.label}>
                <div className="signal-topline"><span className="signal-icon">{signal.icon}</span><span className="signal-label">{signal.label}</span><span className="signal-severity">{signal.severity}</span></div>
                <strong>{signal.title}</strong>
                <p>{signal.detail}</p>
              </div>
            ))}
          </div>
        </section>
      )
    }

    function OverviewResults({ data }) {
      const sentimentBar = Object.entries(data.sentiment_breakdown).map(([name, value]) => ({
        name: name[0].toUpperCase() + name.slice(1), value, color: COLORS[name],
      }))
      const positiveRate = Math.round((data.sentiment_breakdown.positive / Math.max(1, data.total)) * 100)
      const negativeRate = Math.round((data.sentiment_breakdown.negative / Math.max(1, data.total)) * 100)
      const latestVolume = data.volume_by_date.at(-1)?.count || 0
      const previousVolumes = data.volume_by_date.slice(0, -1).map((item) => item.count)
      const averagePreviousVolume = previousVolumes.length ? previousVolumes.reduce((sum, value) => sum + value, 0) / previousVolumes.length : 0
      const coverageSpike = averagePreviousVolume > 0 && latestVolume >= averagePreviousVolume * 1.5
      const toneIsNegative = negativeRate > positiveRate
      const topKeyword = data.top_keywords[0]

      return (
        <div className="results-stack">
          <div className="result-intro">
            <div>
              <span className="eyebrow">Live topic report</span>
              <h1>{data.topic}</h1>
              <p>Signals collected from the latest available headlines{data.category ? ` in ${data.category}` : " across all categories"}.</p>
            </div>
            <div className="report-status"><span className="pulse" /> Updated just now</div>
          </div>

          <div className="metric-grid">
            <div className="metric-card featured"><span className="metric-label">Total coverage</span><strong>{data.total}</strong><small>headlines analysed</small></div>
            <div className="metric-card"><span className="metric-label">Positive signal</span><strong>{positiveRate}%</strong><small>{data.sentiment_breakdown.positive} stories</small></div>
            <div className="metric-card"><span className="metric-label">Negative signal</span><strong>{negativeRate}%</strong><small>{data.sentiment_breakdown.negative} stories</small></div>
          </div>

          <SignalPanel signals={[
            { label: "Tone", icon: toneIsNegative ? "!" : "+", severity: toneIsNegative ? "Watch" : "Positive", tone: toneIsNegative ? "negative" : "positive", title: toneIsNegative ? `${negativeRate}% of coverage is negative` : `${positiveRate}% of coverage is positive`, detail: toneIsNegative ? "Negative stories currently outweigh positive stories." : "Positive stories currently lead the conversation." },
            { label: "Momentum", icon: coverageSpike ? "↑" : "→", severity: coverageSpike ? "Spike" : "Steady", tone: coverageSpike ? "attention" : "neutral", title: coverageSpike ? "Coverage is accelerating" : "Coverage is steady", detail: coverageSpike ? `${latestVolume} headlines on the latest day versus ${Math.round(averagePreviousVolume)} on an average earlier day.` : "No sharp one-day change detected in the available coverage." },
            { label: "Conversation", icon: "#", severity: "Driver", tone: "neutral", title: topKeyword ? `“${topKeyword.word}” is leading` : "No clear keyword driver", detail: topKeyword ? `This keyword appears ${topKeyword.count} times across the analysed headlines.` : "More headlines are needed to identify a recurring topic." },
          ]} />

          <div className="content-grid">
            <section className="panel trend-panel">
              <ChartHeader eyebrow="Momentum" title="Coverage over time" detail="Daily volume" />
              <ResponsiveContainer width="100%" height={245}>
                <LineChart data={data.volume_by_date} margin={{ top: 12, right: 12, left: -18, bottom: 0 }}>
                  <XAxis dataKey="date" tick={{ fill: "#789087", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#789087", fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                  <Tooltip {...chartTooltip} />
                  <Line type="monotone" dataKey="count" stroke="#0f766e" strokeWidth={3} dot={{ r: 3, fill: "#0f766e", strokeWidth: 0 }} activeDot={{ r: 5 }} />
                </LineChart>
              </ResponsiveContainer>
            </section>

            <section className="panel sentiment-panel">
              <ChartHeader eyebrow="Tone" title="Sentiment mix" detail={`${data.total} total`} />
              <ResponsiveContainer width="100%" height={185}>
                <BarChart data={sentimentBar} margin={{ top: 8, right: 4, left: -22, bottom: 0 }}>
                  <XAxis dataKey="name" tick={{ fill: "#789087", fontSize: 11 }} axisLine={false} tickLine={false} />
                  <YAxis tick={{ fill: "#789087", fontSize: 11 }} axisLine={false} tickLine={false} allowDecimals={false} />
                  <Tooltip {...chartTooltip} />
                  <Bar dataKey="value" radius={[5, 5, 0, 0]}>{sentimentBar.map((entry) => <Cell key={entry.name} fill={entry.color} />)}</Bar></BarChart>
                </ResponsiveContainer>
                <SentimentLegend breakdown={data.sentiment_breakdown} />
              </section>
            </div>

            <div className="content-grid lower-grid">
              <section className="panel">
                <ChartHeader eyebrow="Language" title="Top keywords" detail="Most repeated" />
                <div className="keywords">
                  {data.top_keywords.map((keyword) => <span className="keyword-tag" key={keyword.word}>{keyword.word}<b>{keyword.count}</b></span>)}
                </div>
              </section>
              <section className="panel headline-panel">
                <ChartHeader eyebrow="Latest signals" title="Headline feed" detail={`${data.articles?.length || 0} shown`} />
                <HeadlineList articles={data.articles} compact />
              </section>
            </div>
          </div>
      )
    }

    function CompareResults({ data }) {
      const countryEntries = Object.entries(data.countries)
      const strongestMarket = countryEntries.reduce((best, entry) => {
        if (!best || entry[1].sentiment_breakdown.positive > best[1].sentiment_breakdown.positive) return entry
        return best
      }, null)
      const mostNegativeMarket = countryEntries.reduce((worst, entry) => {
        if (!worst || entry[1].sentiment_breakdown.negative > worst[1].sentiment_breakdown.negative) return entry
        return worst
      }, null)

      return (
        <div className="results-stack">
          <div className="result-intro">
            <div><span className="eyebrow">Market comparison</span><h1>{data.topic}</h1><p>Same-size samples make the market signals easier to compare.</p></div>
            <div className="comparison-badge">{Object.keys(data.countries).length} markets</div>
          </div>
          <div className="comparison-summary">
            <div><span className="summary-label">Largest positive signal</span><strong>{strongestMarket ? COUNTRIES.find((item) => item.id === strongestMarket[0])?.short || strongestMarket[0].toUpperCase() : "-"}</strong><small>{strongestMarket ? `${strongestMarket[1].sentiment_breakdown.positive} positive headlines` : "No data"}</small></div>
            <div><span className="summary-label">Comparison method</span><strong>Latest 30</strong><small>headlines per market maximum</small></div>
            <div><span className="summary-label">How to read this</span><strong>Coverage matters</strong><small>Limited samples are marked below</small></div>
          </div>
          <SignalPanel signals={[
            { label: "Positive lead", icon: "+", severity: "Lead", tone: "positive", title: strongestMarket ? `${COUNTRIES.find((item) => item.id === strongestMarket[0])?.short || strongestMarket[0].toUpperCase()} leads positive coverage` : "No positive lead", detail: strongestMarket ? `${strongestMarket[1].sentiment_breakdown.positive} positive headlines in the sampled data.` : "No country has enough data yet." },
            { label: "Negative lead", icon: "!", severity: "Watch", tone: "negative", title: mostNegativeMarket ? `${COUNTRIES.find((item) => item.id === mostNegativeMarket[0])?.short || mostNegativeMarket[0].toUpperCase()} has the most negative coverage` : "No negative lead", detail: mostNegativeMarket ? `${mostNegativeMarket[1].sentiment_breakdown.negative} negative headlines in the sampled data.` : "No country has enough data yet." },
            { label: "Comparison", icon: "↔", severity: "Context", tone: "neutral", title: "Read volume with tone", detail: "A larger headline count means more coverage, not automatically more positive engagement." },
          ]} />
          <div className="country-grid">
            {countryEntries.map(([countryId, countryData]) => {
              const country = COUNTRIES.find((item) => item.id === countryId)
              return (
                <section className="panel country-panel" key={countryId}>
                  <div className="country-heading"><div className="country-code">{country?.short || countryId.toUpperCase()}</div><div><h2>{country?.name || countryId}</h2><span>{countryData.sample_size || 0} of {countryData.sample_limit || 30} headlines sampled</span></div><span className={`coverage-badge${countryData.coverage === "Full sample" ? " full" : " limited"}`}>{countryData.coverage || "Limited coverage"}</span></div>
                  {countryData.total > 0 ? <>
                    <div className="country-chart-block"><div className="mini-chart-title"><span>Coverage over time</span><small>headlines / day</small></div><div className="country-chart"><ResponsiveContainer width="100%" height={145}><LineChart data={countryData.volume_by_date} margin={{ top: 8, right: 6, left: -28, bottom: 0 }}><XAxis dataKey="date" hide /><YAxis hide /><Tooltip {...chartTooltip} /><Line type="monotone" dataKey="count" stroke="#0f766e" strokeWidth={2.5} dot={false} /></LineChart></ResponsiveContainer></div></div>
                    <div className="country-chart-block tone-chart-block"><div className="mini-chart-title"><span>Tone over time</span><small>-1 negative / +1 positive</small></div><div className="country-chart"><ResponsiveContainer width="100%" height={120}><LineChart data={countryData.sentiment_by_date} margin={{ top: 8, right: 6, left: -28, bottom: 0 }}><XAxis dataKey="date" hide /><YAxis domain={[-1, 1]} hide /><Tooltip {...chartTooltip} /><Line type="monotone" dataKey="score" stroke="#d95d39" strokeWidth={2.5} dot={false} /></LineChart></ResponsiveContainer></div></div>
                    <SentimentBar percentages={countryData.sentiment_percentages || { positive: 0, negative: 0, neutral: 0 }} />
                    <SentimentLegend breakdown={countryData.sentiment_breakdown} />
                    <HeadlineList articles={countryData.articles} compact />
                  </> : <p className="empty-copy">No data found for this market.</p>}
                </section>
              )
            })}
          </div>
        </div>
      )
    }

    return (
      <div className="app-shell">
        <header className="topbar"><a className="brand" href="/" onClick={(event) => { event.preventDefault(); setData(null); setCompareData(null) }}><span className="brand-mark">ET</span><span>Engagement Tracker</span></a><span className="topbar-note">Audience intelligence, at a glance</span><span className="live-pill"><span className="pulse" /> Live data</span></header>
        <main>
          <section className="workspace-hero">
            <div className="hero-copy"><span className="eyebrow">Insight workspace</span><h1>Understand what people are saying.</h1><p>Search a topic to turn the latest headlines into a clear read on momentum, tone, and attention.</p></div>
            <div className="search-panel">
              <div className="search-input-wrap"><span className="search-icon">/</span><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === "Enter" && search()} placeholder="Search a brand, topic, or campaign" aria-label="Search topic" /><kbd>Enter</kbd></div>
              <div className="search-options"><label>Category<select value={category} onChange={(event) => setCategory(event.target.value)}><option value="">All categories</option><option value="business">Business</option><option value="technology">Technology</option><option value="entertainment">Entertainment</option><option value="sports">Sports</option><option value="politics">Politics</option></select></label><button className="primary-button" onClick={() => search()} disabled={loading}>{loading ? "Loading..." : "Analyse topic"}<span>{"->"}</span></button></div>
              <div className="compare-row"><span className="option-label">Compare markets</span>{COUNTRIES.map((country) => <label className={`country-option${selectedCountries.includes(country.id) ? " selected" : ""}`} key={country.id} title={`${selectedCountries.includes(country.id) ? "Remove" : "Add"} ${country.name}`}><input type="checkbox" aria-label={country.name} checked={selectedCountries.includes(country.id)} onChange={() => toggleCountry(country.id)} disabled={!selectedCountries.includes(country.id) && selectedCountries.length >= 3} /><span>{country.short}</span></label>)}<span className="selection-count">{selectedCountries.length}/3 selected</span><button className="text-button" onClick={compare} disabled={loading || !selectedCountries.length}>Run comparison <span>{"->"}</span></button></div>
            </div>
          </section>

          <div className="main-layout">
              <aside className="activity-rail"><div className="rail-heading"><span className="eyebrow">Workspace</span><span className="count-badge">{history.length}</span></div><h2>Recent topics</h2>{history.length ? <div className="history-list">{history.map((topic) => <button key={topic} onClick={() => { setQuery(topic); search(topic) }}><span className="topic-dot" />{topic}<span className="arrow">{"->"}</span></button>)}</div> : <p className="rail-empty">Your searched topics will appear here.</p>}<div className="rail-footer"><span className="pulse" /> API connected<br /><small>Ready for a new query</small></div></aside>
              <div className="results-area">{error && <div className="notice error"><strong>Something went wrong</strong><span>{error}</span></div>}{loading && <div className="notice loading"><span className="spinner" /> Pulling the latest signals...</div>}{data && <OverviewResults data={data} />}{compareData && <CompareResults data={compareData} />}{!data && !compareData && !loading && !error && <div className="empty-state"><div className="empty-graphic"><span>+</span><span>+</span><span>+</span></div><span className="eyebrow">Start with a question</span><h2>What deserves your attention?</h2><p>Try a brand, event, or conversation above. Your report will appear here with the signals that matter.</p><div className="suggestions">{["OpenAI", "Nike", "Tesla"].map((suggestion) => <button key={suggestion} onClick={() => { setQuery(suggestion); search(suggestion) }}>{suggestion}<span>{"->"}</span></button>)}</div></div>}</div>
          </div>
        </main>
        <footer className="footer"><span>Engagement Tracker</span><span>Built for sharper decisions</span></footer>
      </div>
    )
}
