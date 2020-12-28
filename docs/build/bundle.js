var app=function(t){"use strict";function e(){}function n(t,e){for(const n in e)t[n]=e[n];return t}function r(t){return t()}function l(){return Object.create(null)}function o(t){t.forEach(r)}function a(t){return"function"==typeof t}function i(t,e){return t!=t?e==e:t!==e||t&&"object"==typeof t||"function"==typeof t}function s(t){const e={};for(const n in t)"$"!==n[0]&&(e[n]=t[n]);return e}function u(t){return null==t?"":t}function c(t,e){t.appendChild(e)}function d(t,e,n){t.insertBefore(e,n||null)}function f(t){t.parentNode.removeChild(t)}function m(t,e){for(let n=0;n<t.length;n+=1)t[n]&&t[n].d(e)}function g(t){return document.createElement(t)}function p(t){return document.createElementNS("http://www.w3.org/2000/svg",t)}function h(t){return document.createTextNode(t)}function $(){return h(" ")}function y(t,e,n,r){return t.addEventListener(e,n,r),()=>t.removeEventListener(e,n,r)}function x(t,e,n){null==n?t.removeAttribute(e):t.getAttribute(e)!==n&&t.setAttribute(e,n)}function b(t,e){e=""+e,t.wholeText!==e&&(t.data=e)}function v(t,e,n,r){t.style.setProperty(e,n,r?"important":"")}function w(t,e){for(let n=0;n<t.options.length;n+=1){const r=t.options[n];if(r.__value===e)return void(r.selected=!0)}}function k(t,e,n){t.classList[n?"add":"remove"](e)}let S;function A(t){S=t}function R(t){(function(){if(!S)throw new Error("Function called outside component initialization");return S})().$$.on_mount.push(t)}const Q=[],E=[],V=[],T=[],_=Promise.resolve();let z=!1;function I(t){V.push(t)}function N(t){T.push(t)}let P=!1;const M=new Set;function Y(){if(!P){P=!0;do{for(let t=0;t<Q.length;t+=1){const e=Q[t];A(e),B(e.$$)}for(A(null),Q.length=0;E.length;)E.pop()();for(let t=0;t<V.length;t+=1){const e=V[t];M.has(e)||(M.add(e),e())}V.length=0}while(Q.length);for(;T.length;)T.pop()();z=!1,P=!1,M.clear()}}function B(t){if(null!==t.fragment){t.update(),o(t.before_update);const e=t.dirty;t.dirty=[-1],t.fragment&&t.fragment.p(t.ctx,e),t.after_update.forEach(I)}}const L=new Set;let F;function X(){F={r:0,c:[],p:F}}function C(){F.r||o(F.c),F=F.p}function G(t,e){t&&t.i&&(L.delete(t),t.i(e))}function O(t,e,n,r){if(t&&t.o){if(L.has(t))return;L.add(t),F.c.push((()=>{L.delete(t),r&&(n&&t.d(1),r())})),t.o(e)}}function j(t,e,n){const r=t.$$.props[e];void 0!==r&&(t.$$.bound[r]=n,n(t.$$.ctx[r]))}function q(t){t&&t.c()}function H(t,e,n){const{fragment:l,on_mount:i,on_destroy:s,after_update:u}=t.$$;l&&l.m(e,n),I((()=>{const e=i.map(r).filter(a);s?s.push(...e):o(e),t.$$.on_mount=[]})),u.forEach(I)}function D(t,e){const n=t.$$;null!==n.fragment&&(o(n.on_destroy),n.fragment&&n.fragment.d(e),n.on_destroy=n.fragment=null,n.ctx=[])}function J(t,e){-1===t.$$.dirty[0]&&(Q.push(t),z||(z=!0,_.then(Y)),t.$$.dirty.fill(0)),t.$$.dirty[e/31|0]|=1<<e%31}function K(t,n,r,a,i,s,u=[-1]){const c=S;A(t);const d=n.props||{},m=t.$$={fragment:null,ctx:null,props:s,update:e,not_equal:i,bound:l(),on_mount:[],on_destroy:[],before_update:[],after_update:[],context:new Map(c?c.$$.context:[]),callbacks:l(),dirty:u,skip_bound:!1};let g=!1;if(m.ctx=r?r(t,d,((e,n,...r)=>{const l=r.length?r[0]:n;return m.ctx&&i(m.ctx[e],m.ctx[e]=l)&&(!m.skip_bound&&m.bound[e]&&m.bound[e](l),g&&J(t,e)),n})):[],m.update(),g=!0,o(m.before_update),m.fragment=!!a&&a(m.ctx),n.target){if(n.hydrate){const t=function(t){return Array.from(t.childNodes)}(n.target);m.fragment&&m.fragment.l(t),t.forEach(f)}else m.fragment&&m.fragment.c();n.intro&&G(t.$$.fragment),H(t,n.target,n.anchor),Y()}A(c)}class U{$destroy(){D(this,1),this.$destroy=e}$on(t,e){const n=this.$$.callbacks[t]||(this.$$.callbacks[t]=[]);return n.push(e),()=>{const t=n.indexOf(e);-1!==t&&n.splice(t,1)}}$set(t){var e;this.$$set&&(e=t,0!==Object.keys(e).length)&&(this.$$.skip_bound=!0,this.$$set(t),this.$$.skip_bound=!1)}}function W(t){let n,r,l;return{c(){n=p("svg"),r=p("path"),x(r,"fill","currentColor"),x(r,"d",t[0]),x(n,"aria-hidden","true"),x(n,"class",l=u(t[1])+" svelte-1d15yci"),x(n,"role","img"),x(n,"xmlns","http://www.w3.org/2000/svg"),x(n,"viewBox",t[2])},m(t,e){d(t,n,e),c(n,r)},p(t,[e]){1&e&&x(r,"d",t[0]),2&e&&l!==(l=u(t[1])+" svelte-1d15yci")&&x(n,"class",l),4&e&&x(n,"viewBox",t[2])},i:e,o:e,d(t){t&&f(n)}}}function Z(t,e,r){let{icon:l}=e,o=[],a="",i="";return t.$$set=t=>{r(4,e=n(n({},e),s(t))),"icon"in t&&r(3,l=t.icon)},t.$$.update=()=>{8&t.$$.dirty&&r(2,i="0 0 "+l.icon[0]+" "+l.icon[1]),r(1,a="fa-svelte "+(e.class?e.class:"")),8&t.$$.dirty&&r(0,o=l.icon[4])},e=s(e),[o,a,i,l]}class tt extends U{constructor(t){super(),K(this,t,Z,W,i,{icon:3})}}
/*!
     * Font Awesome Free 5.15.1 by @fontawesome - https://fontawesome.com
     * License - https://fontawesome.com/license/free (Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT License)
     */var et={prefix:"far",iconName:"arrow-alt-circle-up",icon:[512,512,[],"f35b","M256 504c137 0 248-111 248-248S393 8 256 8 8 119 8 256s111 248 248 248zm0-448c110.5 0 200 89.5 200 200s-89.5 200-200 200S56 366.5 56 256 145.5 56 256 56zm20 328h-40c-6.6 0-12-5.4-12-12V256h-67c-10.7 0-16-12.9-8.5-20.5l99-99c4.7-4.7 12.3-4.7 17 0l99 99c7.6 7.6 2.2 20.5-8.5 20.5h-67v116c0 6.6-5.4 12-12 12z"]};function nt(t){let e,n,r=t[2].toFixed()+"";return{c(){e=g("span"),n=h(r),x(e,"class","svelte-6n71zn")},m(t,r){d(t,e,r),c(e,n)},p(t,e){4&e&&r!==(r=t[2].toFixed()+"")&&b(n,r)},d(t){t&&f(e)}}}function rt(t){let e,n,r=t[2].toFixed(1)+"";return{c(){e=g("span"),n=h(r),x(e,"class","svelte-6n71zn")},m(t,r){d(t,e,r),c(e,n)},p(t,e){4&e&&r!==(r=t[2].toFixed(1)+"")&&b(n,r)},d(t){t&&f(e)}}}function lt(t){let e,n,r;return n=new tt({props:{icon:et}}),{c(){e=g("div"),q(n.$$.fragment),v(e,"font-size","32px"),v(e,"transform","rotate("+t[4]+"deg)"),x(e,"class","svelte-6n71zn")},m(t,l){d(t,e,l),H(n,e,null),r=!0},p(t,n){(!r||16&n)&&v(e,"transform","rotate("+t[4]+"deg)")},i(t){r||(G(n.$$.fragment,t),r=!0)},o(t){O(n.$$.fragment,t),r=!1},d(t){t&&f(e),D(n)}}}function ot(t){let n;return{c(){n=h("terminal state")},m(t,e){d(t,n,e)},p:e,i:e,o:e,d(t){t&&f(n)}}}function at(t){let n;return{c(){n=h("blocked")},m(t,e){d(t,n,e)},p:e,i:e,o:e,d(t){t&&f(n)}}}function it(t){let e,n,r,l,o,a,i,s,u,m,p,y,v,w,S,A,R,Q,E,V,T,_,z,I,N,P,M,Y,B,L,F=t[3][st.up].toFixed(3)+"",j=t[3][st.left].toFixed(3)+"",q=t[3][st.right].toFixed(3)+"",H=t[3][st.down].toFixed(3)+"";function D(t,e){return(null==u||4&e)&&(u=!!(Math.abs(t[2])<10)),u?rt:nt}let J=D(t,-1),K=J(t);const U=[at,ot,lt],W=[];function Z(t,e){return t[0]?0:t[1]?1:2}return A=Z(t),R=W[A]=U[A](t),{c(){e=g("div"),n=g("div"),r=$(),l=g("div"),o=g("span"),a=h(F),i=$(),s=g("div"),K.c(),m=$(),p=g("div"),y=g("span"),v=h(j),w=$(),S=g("div"),R.c(),Q=$(),E=g("div"),V=g("span"),T=h(q),_=$(),z=g("div"),I=$(),N=g("div"),P=g("span"),M=h(H),Y=$(),B=g("div"),x(o,"class","svelte-6n71zn"),x(l,"class","sub-tile sub-tile-top svelte-6n71zn"),x(s,"class","sub-tile sub-tile-reward svelte-6n71zn"),k(s,"reward-is-negative",t[2]<0),x(y,"class","svelte-6n71zn"),x(p,"class","sub-tile sub-tile-left svelte-6n71zn"),x(S,"class","sub-tile sub-tile-middle svelte-6n71zn"),x(V,"class","svelte-6n71zn"),x(E,"class","sub-tile sub-tile-right svelte-6n71zn"),x(P,"class","svelte-6n71zn"),x(N,"class","sub-tile sub-tile-bottom svelte-6n71zn"),x(e,"class","tile svelte-6n71zn"),x(e,"style",t[5]),k(e,"blocked",t[0]),k(e,"terminal",t[1])},m(t,u){d(t,e,u),c(e,n),c(e,r),c(e,l),c(l,o),c(o,a),c(e,i),c(e,s),K.m(s,null),c(e,m),c(e,p),c(p,y),c(y,v),c(e,w),c(e,S),W[A].m(S,null),c(e,Q),c(e,E),c(E,V),c(V,T),c(e,_),c(e,z),c(e,I),c(e,N),c(N,P),c(P,M),c(e,Y),c(e,B),L=!0},p(t,[n]){(!L||8&n)&&F!==(F=t[3][st.up].toFixed(3)+"")&&b(a,F),J===(J=D(t,n))&&K?K.p(t,n):(K.d(1),K=J(t),K&&(K.c(),K.m(s,null))),4&n&&k(s,"reward-is-negative",t[2]<0),(!L||8&n)&&j!==(j=t[3][st.left].toFixed(3)+"")&&b(v,j);let r=A;A=Z(t),A===r?W[A].p(t,n):(X(),O(W[r],1,1,(()=>{W[r]=null})),C(),R=W[A],R?R.p(t,n):(R=W[A]=U[A](t),R.c()),G(R,1),R.m(S,null)),(!L||8&n)&&q!==(q=t[3][st.right].toFixed(3)+"")&&b(T,q),(!L||8&n)&&H!==(H=t[3][st.down].toFixed(3)+"")&&b(M,H),(!L||32&n)&&x(e,"style",t[5]),1&n&&k(e,"blocked",t[0]),2&n&&k(e,"terminal",t[1])},i(t){L||(G(R),L=!0)},o(t){O(R),L=!1},d(t){t&&f(e),K.d(),W[A].d()}}}const st=Object.freeze({up:0,right:1,down:2,left:3});function ut(t,e,n){let r,l,o,a,i=!1,s=!1,u=0,c="";const d=()=>{l=Math.max(...r),o=r.indexOf(l),n(4,a=(90*o).toFixed())},f=()=>{n(3,r=Array(4).fill().map((()=>Math.random()))),d()};return f(),[i,s,u,r,a,c,()=>i,()=>{n(0,i=!0)},()=>s,()=>{n(1,s=!0)},()=>u,t=>{n(2,u=t)},()=>o,t=>s?u:r[t],()=>s?u:l,(t,e)=>{n(3,r[t]=e,r),d()},f,t=>{let e,r,l,o=.35+.6*t;i||(e=0+255*o,r=110+145*o,l=210+45*o,n(5,c="background: rgb("+e+","+r+","+l+", 1);"))}]}class ct extends U{constructor(t){super(),K(this,t,ut,it,i,{isBlocked:6,setBlocked:7,isTerminal:8,setTerminal:9,getReward:10,setReward:11,getPolicy:12,getQValue:13,getMaxQValue:14,setQValue:15,initQValues:16,setHeat:17})}get isBlocked(){return this.$$.ctx[6]}get setBlocked(){return this.$$.ctx[7]}get isTerminal(){return this.$$.ctx[8]}get setTerminal(){return this.$$.ctx[9]}get getReward(){return this.$$.ctx[10]}get setReward(){return this.$$.ctx[11]}get getPolicy(){return this.$$.ctx[12]}get getQValue(){return this.$$.ctx[13]}get getMaxQValue(){return this.$$.ctx[14]}get setQValue(){return this.$$.ctx[15]}get initQValues(){return this.$$.ctx[16]}get setHeat(){return this.$$.ctx[17]}}function dt(t,e,n){const r=t.slice();return r[21]=e[n],r[22]=e,r[23]=n,r}function ft(t,e,n){const r=t.slice();return r[21]=e[n],r[24]=e,r[25]=n,r}function mt(t){let e,n,r=t[25],l=t[23];const o=()=>t[18](e,r,l),a=()=>t[18](null,r,l);return e=new ct({props:{}}),o(),{c(){q(e.$$.fragment)},m(t,r){H(e,t,r),n=!0},p(t,n){r===t[25]&&l===t[23]||(a(),r=t[25],l=t[23],o());e.$set({})},i(t){n||(G(e.$$.fragment,t),n=!0)},o(t){O(e.$$.fragment,t),n=!1},d(t){a(),D(e,t)}}}function gt(t){let e,n,r=Array(t[0]),l=[];for(let e=0;e<r.length;e+=1)l[e]=mt(ft(t,r,e));const o=t=>O(l[t],1,1,(()=>{l[t]=null}));return{c(){for(let t=0;t<l.length;t+=1)l[t].c();e=h("")},m(t,r){for(let e=0;e<l.length;e+=1)l[e].m(t,r);d(t,e,r),n=!0},p(t,n){if(5&n){let a;for(r=Array(t[0]),a=0;a<r.length;a+=1){const o=ft(t,r,a);l[a]?(l[a].p(o,n),G(l[a],1)):(l[a]=mt(o),l[a].c(),G(l[a],1),l[a].m(e.parentNode,e))}for(X(),a=r.length;a<l.length;a+=1)o(a);C()}},i(t){if(!n){for(let t=0;t<r.length;t+=1)G(l[t]);n=!0}},o(t){l=l.filter(Boolean);for(let t=0;t<l.length;t+=1)O(l[t]);n=!1},d(t){m(l,t),t&&f(e)}}}function pt(t){let e,n,r=Array(t[1]),l=[];for(let e=0;e<r.length;e+=1)l[e]=gt(dt(t,r,e));const o=t=>O(l[t],1,1,(()=>{l[t]=null}));return{c(){e=g("div");for(let t=0;t<l.length;t+=1)l[t].c();x(e,"class","maze svelte-csffwp"),v(e,"grid-template-columns","repeat("+t[0]+", 100px)")},m(t,r){d(t,e,r);for(let t=0;t<l.length;t+=1)l[t].m(e,null);n=!0},p(t,[a]){if(7&a){let n;for(r=Array(t[1]),n=0;n<r.length;n+=1){const o=dt(t,r,n);l[n]?(l[n].p(o,a),G(l[n],1)):(l[n]=gt(o),l[n].c(),G(l[n],1),l[n].m(e,null))}for(X(),n=r.length;n<l.length;n+=1)o(n);C()}(!n||1&a)&&v(e,"grid-template-columns","repeat("+t[0]+", 100px)")},i(t){if(!n){for(let t=0;t<r.length;t+=1)G(l[t]);n=!0}},o(t){l=l.filter(Boolean);for(let t=0;t<l.length;t+=1)O(l[t]);n=!1},d(t){t&&f(e),m(l,t)}}}function ht(t,e,n){let{numX:r}=e,{numY:l}=e,{blocked:o=Array()}=e,{terminal:a=Array()}=e,{rewards:i=Array()}=e,{defaultReward:s=0}=e,u=Array.from({length:r},(()=>Array.from({length:l},(()=>null))));R((()=>{o.forEach((t=>{u[t[0]][t[1]].setBlocked()})),a.forEach((t=>{u[t[0]][t[1]].setTerminal()}));for(let t=0;t<l;t++)for(let e=0;e<r;e++)u[e][t].setReward(s);i.forEach((t=>{u[t[0]][t[1]].setReward(t[2])})),m()}));const c=t=>Math.floor(Math.random()*Math.floor(t)),d=(t,e)=>u[t][e].isBlocked(),f=(t,e)=>u[t][e].isTerminal(),m=()=>{for(let t=0;t<l;t++)for(let e=0;e<r;e++)u[e][t].initQValues();h()},g=(t,e)=>u[t][e].getPolicy(),p=(t,e)=>t>=0&&t<r&&e>=0&&e<l?u[t][e].getReward():(console.log("ERROR: Invalid getReward coordinates [",t,":",e,"] !"),0),h=()=>{let t=1e6,e=-1e6;for(let n=0;n<l;n++)for(let l=0;l<r;l++){let r=u[l][n].getMaxQValue();t>r&&(t=r),e<r&&(e=r)}let n=e-t;for(let e=0;e<l;e++)for(let l=0;l<r;l++){let r=u[l][e].getMaxQValue();u[l][e].setHeat((r-t)/n)}};return t.$$set=t=>{"numX"in t&&n(0,r=t.numX),"numY"in t&&n(1,l=t.numY),"blocked"in t&&n(3,o=t.blocked),"terminal"in t&&n(4,a=t.terminal),"rewards"in t&&n(5,i=t.rewards),"defaultReward"in t&&n(6,s=t.defaultReward)},[r,l,u,o,a,i,s,c,d,f,m,(t,e,n,o)=>{t>=0&&t<r&&e>=0&&e<l?(u[t][e].setQValue(n,o),h()):console.log("ERROR: Invalid setQValue coordinates [",t,":",e,"] !")},(t,e,n)=>u[t][e].getQValue(n),(t,e)=>u[t][e].getMaxQValue(),g,()=>{for(;;){let t=c(r),e=c(l);if(!f(t,e)&&!d(t,e))return[t,e]}},(t,e,n)=>Math.random()<n?c(4):g(t,e),(t,e,n)=>{let o=Number(t),a=Number(e);return n==st.down&&e<l-1&&(a+=1),n==st.right&&t<r-1&&(o+=1),n==st.up&&e>0&&(a-=1),n==st.left&&t>0&&(o-=1),d(o,a)&&(o=Number(t),a=Number(e)),[o,a,p(o,a)]},function(t,e,r){E[t?"unshift":"push"]((()=>{u[e][r]=t,n(2,u)}))}]}class $t extends U{constructor(t){super(),K(this,t,ht,pt,i,{numX:0,numY:1,blocked:3,terminal:4,rewards:5,defaultReward:6,getRandomInt:7,isBlocked:8,isTerminal:9,initQValues:10,setQValue:11,getQValue:12,getMaxQValue:13,getPolicy:14,getRandomStartState:15,getEpsilonGreedyAction:16,step:17})}get getRandomInt(){return this.$$.ctx[7]}get isBlocked(){return this.$$.ctx[8]}get isTerminal(){return this.$$.ctx[9]}get initQValues(){return this.$$.ctx[10]}get setQValue(){return this.$$.ctx[11]}get getQValue(){return this.$$.ctx[12]}get getMaxQValue(){return this.$$.ctx[13]}get getPolicy(){return this.$$.ctx[14]}get getRandomStartState(){return this.$$.ctx[15]}get getEpsilonGreedyAction(){return this.$$.ctx[16]}get step(){return this.$$.ctx[17]}}function yt(t){let n;return{c(){n=g("div"),x(n,"class","plot svelte-1qi4jkb")},m(e,r){d(e,n,r),t[12](n)},p:e,i:e,o:e,d(e){e&&f(n),t[12](null)}}}function xt(t,e,n){let r,{title:l=""}=e,{xIsLog:o=!1}=e,{data:a=Array()}=e,{yIsLog:i=!1}=e,{yTitle:s=""}=e,{hasSecondY:u=!1}=e,{dataSecond:c=Array()}=e,{ySecondIsLog:d=!1}=e,{ySecondTitle:f=""}=e;const m=()=>{let t={title:l,showlegend:!1,xaxis:{type:o?"log":"linear",autorange:!0},yaxis:{title:s,type:i?"log":"linear",titlefont:{color:"#08C"},tickfont:{color:"#08C"},autorange:!0},margin:{autoexpand:!1,t:50,l:40,b:30,r:20}};""!=s&&(t.margin.l=60);let e=[{x:[...Array(a.length).keys()],y:a,mode:"lines",line:{shape:"spline"},type:"scatter"}];if(u){t.yaxis2={title:f,type:d?"log":"linear",overlaying:"y",side:"right",titlefont:{color:"#E60"},tickfont:{color:"#E60"},autorange:!0},t.margin.r=""!=f?60:40;let n={x:[...Array(c.length).keys()],y:c,mode:"lines",line:{shape:"spline"},type:"scatter",yaxis:"y2"};e.push(n)}Plotly.react(r,e,t,{displaylogo:!1})};return R((()=>{m()})),t.$$set=t=>{"title"in t&&n(3,l=t.title),"xIsLog"in t&&n(4,o=t.xIsLog),"data"in t&&n(1,a=t.data),"yIsLog"in t&&n(5,i=t.yIsLog),"yTitle"in t&&n(6,s=t.yTitle),"hasSecondY"in t&&n(7,u=t.hasSecondY),"dataSecond"in t&&n(2,c=t.dataSecond),"ySecondIsLog"in t&&n(8,d=t.ySecondIsLog),"ySecondTitle"in t&&n(9,f=t.ySecondTitle)},[r,a,c,l,o,i,s,u,d,f,m,()=>{n(1,a=Array()),n(2,c=Array()),m()},function(t){E[t?"unshift":"push"]((()=>{r=t,n(0,r)}))}]}class bt extends U{constructor(t){super(),K(this,t,xt,yt,i,{title:3,xIsLog:4,data:1,yIsLog:5,yTitle:6,hasSecondY:7,dataSecond:2,ySecondIsLog:8,ySecondTitle:9,updatePlot:10,clearPlot:11})}get updatePlot(){return this.$$.ctx[10]}get clearPlot(){return this.$$.ctx[11]}}function vt(t,e,n){const r=t.slice();return r[47]=e[n],r}function wt(t){let n,r,l,o=t[47].name+"";return{c(){n=g("option"),r=h(o),n.__value=l=t[47].name,n.value=n.__value},m(t,e){d(t,n,e),c(n,r)},p:e,d(t){t&&f(n)}}}function kt(t){let e,n,r,l,a,i,s,u,p,k,S,A,R,Q,V,T,_,z,P,M,Y,B,L,F;function X(e){t[20].call(null,e)}function C(e){t[21].call(null,e)}let J={yTitle:"reward per episode",ySecondTitle:"steps per episode",hasSecondY:!1};void 0!==t[9]&&(J.data=t[9]),void 0!==t[8]&&(J.dataSecond=t[8]),r=new bt({props:J}),t[19](r),E.push((()=>j(r,"data",X))),E.push((()=>j(r,"dataSecond",C)));let K=t[12],U=[];for(let e=0;e<K.length;e+=1)U[e]=wt(vt(t,K,e));let W={numX:t[4],numY:t[5],blocked:t[0],terminal:t[1],rewards:t[2],defaultReward:t[3]};return Y=new $t({props:W}),t[23](Y),{c(){e=g("div"),n=g("div"),q(r.$$.fragment),i=$(),s=g("div"),u=h("EPISODE : "),p=h(t[11]),k=$(),S=g("div"),A=g("select");for(let t=0;t<U.length;t+=1)U[t].c();R=$(),Q=g("button"),Q.textContent="init",V=$(),T=g("button"),T.textContent="halt",_=$(),z=g("button"),z.textContent="run",P=$(),M=g("div"),q(Y.$$.fragment),x(n,"class","narrow-box svelte-ikbdek"),x(s,"class","box svelte-ikbdek"),x(A,"class","svelte-ikbdek"),void 0===t[10]&&I((()=>t[22].call(A))),x(Q,"class","svelte-ikbdek"),x(T,"class","svelte-ikbdek"),x(z,"class","svelte-ikbdek"),x(S,"class","box svelte-ikbdek"),x(M,"class","narrow-box svelte-ikbdek"),x(e,"class","container svelte-ikbdek"),v(e,"width",16+100*t[4]+4*(t[4]-1)+"px")},m(l,o){d(l,e,o),c(e,n),H(r,n,null),c(e,i),c(e,s),c(s,u),c(s,p),c(e,k),c(e,S),c(S,A);for(let t=0;t<U.length;t+=1)U[t].m(A,null);w(A,t[10]),c(S,R),c(S,Q),c(S,V),c(S,T),c(S,_),c(S,z),c(e,P),c(e,M),H(Y,M,null),B=!0,L||(F=[y(A,"change",t[22]),y(A,"change",t[15]),y(Q,"click",t[15]),y(T,"click",t[14]),y(z,"click",t[13])],L=!0)},p(t,n){const o={};if(!l&&512&n[0]&&(l=!0,o.data=t[9],N((()=>l=!1))),!a&&256&n[0]&&(a=!0,o.dataSecond=t[8],N((()=>a=!1))),r.$set(o),(!B||2048&n[0])&&b(p,t[11]),4096&n[0]){let e;for(K=t[12],e=0;e<K.length;e+=1){const r=vt(t,K,e);U[e]?U[e].p(r,n):(U[e]=wt(r),U[e].c(),U[e].m(A,null))}for(;e<U.length;e+=1)U[e].d(1);U.length=K.length}5120&n[0]&&w(A,t[10]);const i={};16&n[0]&&(i.numX=t[4]),32&n[0]&&(i.numY=t[5]),1&n[0]&&(i.blocked=t[0]),2&n[0]&&(i.terminal=t[1]),4&n[0]&&(i.rewards=t[2]),8&n[0]&&(i.defaultReward=t[3]),Y.$set(i),(!B||16&n[0])&&v(e,"width",16+100*t[4]+4*(t[4]-1)+"px")},i(t){B||(G(r.$$.fragment,t),G(Y.$$.fragment,t),B=!0)},o(t){O(r.$$.fragment,t),O(Y.$$.fragment,t),B=!1},d(n){n&&f(e),t[19](null),D(r),m(U,n),t[23](null),D(Y),L=!1,o(F)}}}const St=.1,At=.2,Rt=.9;function Qt(t,e,n){let r,l,o,a,i,{blocked:s=Array()}=e,{terminal:u=Array([0,0])}=e,{rewards:c=Array([0,0,1])}=e,{defaultReward:d=0}=e,{startState:f}=e,{numEpisodes:m=1e3}=e,{planningSteps:g=10}=e,{numX:p=5}=e,{numY:h=5}=e,$=Array(),y=Array(),x=0,b=0,v=0;const w=(t,e,n,l,o,a)=>{let i;i=r.isTerminal(o,a)?l:l+Rt*r.getMaxQValue(o,a);let s=.8*r.getQValue(t,e,n)+At*i;r.setQValue(t,e,n,s)},k=(t,e)=>{let n,o,a,s;r.isTerminal(t,e)?($.push(v),y.push(x),l.updatePlot(),N()):i=setTimeout((()=>{a=r.getEpsilonGreedyAction(t,e,St),[n,o,s]=r.step(t,e,a),w(t,e,a,s,n,o),t=Number(n),e=Number(o),x+=s,v++,k(t,e)}),0)},S=(t,e,n)=>{let o,a,s,u;r.isTerminal(t,e)?($.push(v),y.push(x),l.updatePlot(),N()):i=setTimeout((()=>{[o,a,u]=r.step(t,e,n),s=r.getEpsilonGreedyAction(o,a,St),((t,e,n,l,o,a,i)=>{let s;s=r.isTerminal(o,a)?l:l+Rt*r.getQValue(o,a,i);let u=.8*r.getQValue(t,e,n)+At*s;r.setQValue(t,e,n,u)})(t,e,n,u,o,a,s),t=Number(o),e=Number(a),n=Number(s),x+=u,v++,S(t,e,n)}),0)},A=(t,e,n)=>{let o,a,s,u;r.isTerminal(t,e)?($.push(v),y.push(x),l.updatePlot(),N()):i=setTimeout((()=>{[o,a,u]=r.step(t,e,n),s=r.getEpsilonGreedyAction(o,a,St),((t,e,n,l,o,a,i)=>{let s;if(r.isTerminal(o,a))s=l;else{let t,e=0;t=.025;for(let n=0;n<4;n++)e+=t*r.getQValue(o,a,n);t=.9;let n=r.getPolicy(o,a);e+=t*r.getQValue(o,a,n),s=l+Rt*e}let u=.8*r.getQValue(t,e,n)+At*s;r.setQValue(t,e,n,u)})(t,e,n,u,o,a),t=Number(o),e=Number(a),n=Number(s),x+=u,v++,A(t,e,n)}),0)};let R=Array.from({length:p},(()=>Array.from({length:h},(()=>Array.from({length:4},(()=>null)))))),Q=Array();const V=()=>{let t=r.getRandomInt(Q.length);return Q[t]},T=(t,e)=>{let n,o,a,s;r.isTerminal(t,e)?($.push(v),y.push(x),l.updatePlot(),N()):i=setTimeout((()=>{a=r.getEpsilonGreedyAction(t,e,St),[n,o,s]=r.step(t,e,a),w(t,e,a,s,n,o),((t,e,n,r,l,o)=>{let a=!1;for(let r=0;r<Q.length;r++)if(Q[r][0]==t&&Q[r][1]==e&&Q[r][2]==n){a=!0;break}a||Q.push([t,e,n]),R[t][e][n]=[l,o,r]})(t,e,a,s,n,o),t=Number(n),e=Number(o);for(let t=0;t<g;t++){let t,e,n,r,l,o;[t,e,n]=V(),[l,o,r]=R[t][e][n],w(t,e,n,r,l,o)}x+=s,v++,T(t,e)}),0)};let _;const z=(t,e)=>{let n,o,a,s;r.isTerminal(t,e)?((()=>{for(let t=0;t<h;t++)for(let e=0;e<p;e++)for(let n=0;n<4;n++){let l=!1,o=0,a=1;for(let r=0;r<_.length;r++)e==_[r][0]&&t==_[r][1]&&n==_[r][2]&&(l=!0),l&&(o+=a*_[r][3],a*=Rt);if(l){let l=.8*r.getQValue(e,t,n)+At*o;r.setQValue(e,t,n,l)}}})(),$.push(v),y.push(x),l.updatePlot(),N()):i=setTimeout((()=>{a=r.getEpsilonGreedyAction(t,e,St),[n,o,s]=r.step(t,e,a),_.push([t,e,a,s]),t=Number(n),e=Number(o),x+=s,v++,z(t,e)}),0)},I=[{name:"Q-Learning",func:()=>{let t,e;x=0,v=0,[t,e]=f||r.getRandomStartState(),k(t,e)}},{name:"SARSA",func:()=>{let t,e,n;x=0,v=0,[t,e]=f||r.getRandomStartState(),n=r.getEpsilonGreedyAction(t,e,St),S(t,e,n)}},{name:"Expected SARSA",func:()=>{let t,e,n;x=0,v=0,[t,e]=f||r.getRandomStartState(),n=r.getEpsilonGreedyAction(t,e,St),A(t,e,n)}},{name:"Dyna-Q",func:()=>{let t,e;x=0,v=0,[t,e]=f||r.getRandomStartState(),T(t,e)}},{name:"Monte Carlo",func:()=>{let t,e;x=0,v=0,_=Array(),[t,e]=r.getRandomStartState(),z(t,e)}}],N=()=>{a=setTimeout((()=>{b<m&&(n(11,b++,b),I.forEach((t=>{o==t.name&&t.func()})))}),0)};return t.$$set=t=>{"blocked"in t&&n(0,s=t.blocked),"terminal"in t&&n(1,u=t.terminal),"rewards"in t&&n(2,c=t.rewards),"defaultReward"in t&&n(3,d=t.defaultReward),"startState"in t&&n(16,f=t.startState),"numEpisodes"in t&&n(17,m=t.numEpisodes),"planningSteps"in t&&n(18,g=t.planningSteps),"numX"in t&&n(4,p=t.numX),"numY"in t&&n(5,h=t.numY)},[s,u,c,d,p,h,r,l,$,y,o,b,I,N,()=>{a&&clearTimeout(a),i&&clearTimeout(i)},()=>{n(11,b=0),n(8,$=Array()),n(9,y=Array()),r.initQValues(),l.clearPlot()},f,m,g,function(t){E[t?"unshift":"push"]((()=>{l=t,n(7,l)}))},function(t){y=t,n(9,y)},function(t){$=t,n(8,$)},function(){o=function(t){const e=t.querySelector(":checked")||t.options[0];return e&&e.__value}(this),n(10,o),n(12,I)},function(t){E[t?"unshift":"push"]((()=>{r=t,n(6,r)}))}]}class Et extends U{constructor(t){super(),K(this,t,Qt,kt,i,{blocked:0,terminal:1,rewards:2,defaultReward:3,startState:16,numEpisodes:17,planningSteps:18,numX:4,numY:5},[-1,-1])}}const Vt=document.getElementById("maze-shell-0"),Tt=document.getElementById("maze-shell-1"),_t=document.getElementById("maze-shell-2"),zt=document.getElementById("maze-shell-3"),It=Vt?new Et({target:Vt,props:{numX:12,numY:3,planningSteps:10,blocked:Array(),terminal:Array([1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],[9,2],[10,2],[11,2]),rewards:Array([1,2,-10],[2,2,-10],[3,2,-10],[4,2,-10],[5,2,-10],[6,2,-10],[7,2,-10],[8,2,-10],[9,2,-10],[10,2,-10],[11,2,0]),defaultReward:-.1,startState:[0,2]}}):null,Nt=Tt?new Et({target:Tt,props:{numX:7,numY:3,planningSteps:10,blocked:Array(),terminal:Array([1,2],[2,2],[3,2],[4,2],[5,2],[6,2]),rewards:Array([1,2,-10],[2,2,-10],[3,2,-10],[4,2,-10],[5,2,-10],[6,2,0]),defaultReward:-.1,startState:[0,2]}}):null,Pt=_t?new Et({target:_t,props:{numX:5,numY:5,blocked:Array(),terminal:Array([1,1]),rewards:Array([1,1,1]),defaultReward:-.1}}):null,Mt=zt?new Et({target:zt,props:{numX:6,numY:4,blocked:Array([0,0],[3,2],[4,1],[3,3]),terminal:Array([1,1]),rewards:Array([1,1,1]),defaultReward:-.1}}):null;return t.comp0=It,t.comp1=Nt,t.comp2=Pt,t.comp3=Mt,Object.defineProperty(t,"__esModule",{value:!0}),t}({});
//# sourceMappingURL=bundle.js.map
