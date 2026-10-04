"use client";
import { useEffect, useRef, useState } from "react";

function useChartWidth() {
  const ref=useRef<HTMLElement>(null), [width,setWidth]=useState(850);
  useEffect(()=>{const node=ref.current;if(!node)return;const observer=new ResizeObserver(entries=>setWidth(Math.max(280,entries[0].contentRect.width)));observer.observe(node);return()=>observer.disconnect();},[]);
  return {ref,width};
}

export const number = (value: number | null | undefined, digits = 0) => value == null ? "Unavailable" : value.toLocaleString("en-PH", { maximumFractionDigits: digits, minimumFractionDigits: digits });
export const month = (value: string) => new Date(`${value.slice(0, 10)}T00:00:00`).toLocaleDateString("en-PH", { month: "short", year: "numeric" });
const compact = (value: number) => Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
const colors = ["#006C67", "#B55026", "#596887", "#9B682A"];

type Point = { label: string; values: (number | null)[] };
export function LineChart({ data, names, unit = "arrivals" }: { data: Point[]; names: string[]; unit?: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const {ref,width}=useChartWidth();
  if (!data.length) return null;
  const W = width, H = 310, left = 60, right = 20, top = 25, bottom = 42;
  const ticks=Math.min(data.length,W<520?3:6);
  const values = data.flatMap(p => p.values.filter((v): v is number => v !== null && Number.isFinite(v)));
  const max = Math.max(...values, 1) * 1.08;
  const min = Math.min(0, ...values);
  const x = (i: number) => left + i * (W-left-right) / Math.max(1, data.length-1);
  const y = (v: number) => H-bottom - (v-min) / (max-min) * (H-top-bottom);
  const path = (series: number) => {
    let previous = false;
    return data.map((p, i) => {
      const v = p.values[series];
      if (v == null) { previous = false; return ""; }
      const command = previous ? "L" : "M";
      previous = true;
      return `${command}${x(i)},${y(v)}`;
    }).join(" ");
  };
  const index = Math.min(hover ?? data.length-1, data.length-1);
  return <figure ref={ref} className="chart-figure" tabIndex={0} aria-label={`${names.join(", ")} over time. Use left and right arrow keys to inspect readings.`}
    onKeyDown={e => { if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); setHover(Math.max(0, Math.min(data.length-1, index+(e.key === "ArrowRight" ? 1 : -1)))); } }}>
    <figcaption className="chart-readout"><span>{data[index].label}</span><div>{names.map((n, i) => <span key={n}><i style={{ background: colors[i] }} />{n} <strong>{number(data[index].values[i], unit === "scaled MSE" ? 4 : 0)}</strong></span>)}</div></figcaption>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Line chart in ${unit}`}
      onPointerMove={e => { const box = e.currentTarget.getBoundingClientRect(); const px = (e.clientX-box.left)*W/box.width; setHover(Math.max(0,Math.min(data.length-1,Math.round((px-left)/(W-left-right)*(data.length-1))))); }}
      onPointerLeave={() => setHover(null)}>
      <title>{names.join(", ")} in {unit}</title>
      {Array.from({ length: 5 }, (_, i) => { const v = min + (max-min)*i/4; return <g key={i}><line x1={left} x2={W-right} y1={y(v)} y2={y(v)} stroke="#DFE4DF" strokeDasharray="3 5" /><text x={left-12} y={y(v)+4} textAnchor="end">{unit === "scaled MSE" ? v.toFixed(3) : compact(v)}</text></g>; })}
      {names.map((n,i) => <path key={n} d={path(i)} fill="none" stroke={colors[i]} strokeWidth={i === 0 ? 2.5 : 1.8} strokeDasharray={i > 1 ? "6 4" : undefined} />)}
      {Array.from({ length: ticks }, (_, i) => { const idx = Math.round(i * (data.length-1) / Math.max(1,ticks-1)); return <text key={i} x={x(idx)} y={H-12} textAnchor={i===0?"start":i===ticks-1?"end":"middle"}>{data[idx].label}</text>; })}
      {hover !== null && <g><line x1={x(index)} x2={x(index)} y1={top} y2={H-bottom} stroke="#607583" strokeDasharray="3 3" />{data[index].values.map((v,i) => v !== null && <circle key={i} cx={x(index)} cy={y(v)} r={4} fill={colors[i]} stroke="white" strokeWidth={2} />)}</g>}
    </svg>
    <div className="chart-caption">{unit} · {data.length} observations · Move over the chart or use the arrow keys to inspect values.</div>
  </figure>;
}

export function Importance({ rows, labels }: { rows: { feature: string; value: number }[]; labels: Record<string,string> }) {
  const max = Math.max(...rows.map(r => r.value), 1);
  return <div className="importance" role="img" aria-label="Average absolute SHAP contribution by feature">{rows.map(r => <div className="importance-row" key={r.feature}><span>{labels[r.feature] ?? r.feature}</span><div><i style={{ width: `${r.value/max*100}%` }} /></div><strong>{number(r.value,1)}</strong></div>)}</div>;
}

export function Waterfall({ base, prediction, rows, labels }: { base: number; prediction: number; rows: {feature:string;value:number}[]; labels:Record<string,string> }) {
  const {ref,width}=useChartWidth();
  let running=base;
  const bars=[{name:"Base value",start:0,end:base,value:base,total:true},...rows.map(r=>{const start=running;running+=r.value;return{name:labels[r.feature]??r.feature,start,end:running,value:r.value,total:false};}),{name:"Prediction",start:0,end:prediction,value:prediction,total:true}];
  const W=width,left=W<500?125:175,right=90,top=15,rowHeight=40,H=bars.length*rowHeight+50;
  const min=Math.min(0,...bars.flatMap(b=>[b.start,b.end])), max=Math.max(1,...bars.flatMap(b=>[b.start,b.end]))*1.05;
  const x=(v:number)=>left+(v-min)/(max-min)*(W-left-right);
  const y=(i:number)=>top+i*rowHeight;
  const ticks=W<500?3:5;
  return <figure ref={ref} className="waterfall"><svg style={{minWidth:0}} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="SHAP waterfall from the base value to the prediction"><title>Base plus signed feature contributions equals prediction</title>
    {Array.from({length:ticks},(_,i)=>{const v=min+(max-min)*i/(ticks-1);return <g key={i}><line x1={x(v)} x2={x(v)} y1={top-5} y2={H-38} stroke="#DFE4DF" strokeDasharray="3 5"/><text x={x(v)} y={H-16} textAnchor="middle">{compact(v)}</text></g>;})}
    {bars.map((b,i)=><g key={i}>
      <text x={left-12} y={y(i)+14} textAnchor="end">{b.name.split(" (")[0]}</text>
      <rect x={Math.min(x(b.start),x(b.end))} y={y(i)} width={Math.max(1,Math.abs(x(b.start)-x(b.end)))} height={22} fill={b.total?"#193746":b.value>=0?"#006C67":"#B55026"}/>
      {i<bars.length-1&&<line x1={x(b.end)} x2={x(b.end)} y1={y(i)+22} y2={y(i+1)} stroke="#71828A" strokeDasharray="3 3"/>}
      <text x={W-5} y={y(i)+14} textAnchor="end">{!b.total&&b.value>0?"+":""}{number(b.value)}</text>
      <title>{b.name}: {b.value.toLocaleString()} arrivals</title>
    </g>)}
  </svg><figcaption className="chart-caption">Read from top to bottom. Teal adds; rust subtracts. Feature contributions include all lookback months. The final bar is the model's prediction.</figcaption></figure>;
}

export function Scatter({ rows, label }: { rows:{value:number;shap:number;date:string}[]; label:string }) {
  const {ref,width}=useChartWidth();
  const W=width,H=300,left=65,right=25,top=25,bottom=50;
  const xs=rows.map(r=>r.value),ys=rows.map(r=>r.shap);
  const lo=Math.min(...xs), hi=Math.max(...xs), yl=Math.min(0,...ys),yh=Math.max(1,...ys);
  const x=(v:number)=>left+(v-lo)/Math.max(hi-lo,1e-8)*(W-left-right),y=(v:number)=>H-bottom-(v-yl)/(yh-yl)*(H-top-bottom);
  return <figure ref={ref} className="chart-figure"><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Dependence of the latest ${label} reading and its SHAP contribution`}><title>{label} versus most recent lag SHAP value</title>
    {[0,.25,.5,.75,1].map(p=>{const v=yl+(yh-yl)*p;return <g key={p}><line x1={left} x2={W-right} y1={y(v)} y2={y(v)} stroke="#DFE4DF"/><text x={left-10} y={y(v)+4} textAnchor="end">{compact(v)}</text></g>;})}
    <line x1={left} x2={W-right} y1={y(0)} y2={y(0)} stroke="#71828A" strokeDasharray="5 5"/>
    {[0,.25,.5,.75,1].map(p=>{const v=lo+(hi-lo)*p;return <text key={p} x={x(v)} y={H-bottom+24} textAnchor="middle">{number(v,1)}</text>;})}
    {rows.map((r,i)=><circle key={i} cx={x(r.value)} cy={y(r.shap)} r={5} fill="#006C67" fillOpacity={.72}><title>{month(r.date)}: input {r.value}, SHAP {number(r.shap,1)} arrivals</title></circle>)}
  </svg><figcaption className="chart-caption">Horizontal: {label}. Vertical: SHAP contribution in arrivals, for the most recent input month only.</figcaption></figure>;
}
