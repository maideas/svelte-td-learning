var app=function(t){"use strict";function e(){}function n(t,e){for(const n in e)t[n]=e[n];return t}function r(t){return t()}function l(){return Object.create(null)}function a(t){t.forEach(r)}function s(t){return"function"==typeof t}function o(t,e){return t!=t?e==e:t!==e||t&&"object"==typeof t||"function"==typeof t}function i(t){const e={};for(const n in t)"$"!==n[0]&&(e[n]=t[n]);return e}function u(t){return null==t?"":t}function c(t,e){t.appendChild(e)}function d(t,e,n){t.insertBefore(e,n||null)}function f(t){t.parentNode.removeChild(t)}function g(t,e){for(let n=0;n<t.length;n+=1)t[n]&&t[n].d(e)}function p(t){return document.createElement(t)}function m(t){return document.createElementNS("http://www.w3.org/2000/svg",t)}function h(t){return document.createTextNode(t)}function $(){return h(" ")}function y(t,e,n,r){return t.addEventListener(e,n,r),()=>t.removeEventListener(e,n,r)}function x(t,e,n){null==n?t.removeAttribute(e):t.getAttribute(e)!==n&&t.setAttribute(e,n)}function v(t,e){e=""+e,t.wholeText!==e&&(t.data=e)}function w(t,e,n,r){t.style.setProperty(e,n,r?"important":"")}function b(t,e){for(let n=0;n<t.options.length;n+=1){const r=t.options[n];if(r.__value===e)return void(r.selected=!0)}}function k(t,e,n){t.classList[n?"add":"remove"](e)}let S;function A(t){S=t}function Q(t){(function(){if(!S)throw new Error("Function called outside component initialization");return S})().$$.on_mount.push(t)}const E=[],R=[],V=[],N=[],T=Promise.resolve();let z=!1;function B(t){V.push(t)}function _(t){N.push(t)}let M=!1;const I=new Set;function P(){if(!M){M=!0;do{for(let t=0;t<E.length;t+=1){const e=E[t];A(e),Y(e.$$)}for(A(null),E.length=0;R.length;)R.pop()();for(let t=0;t<V.length;t+=1){const e=V[t];I.has(e)||(I.add(e),e())}V.length=0}while(E.length);for(;N.length;)N.pop()();z=!1,M=!1,I.clear()}}function Y(t){if(null!==t.fragment){t.update(),a(t.before_update);const e=t.dirty;t.dirty=[-1],t.fragment&&t.fragment.p(t.ctx,e),t.after_update.forEach(B)}}const L=new Set;let X;function F(){X={r:0,c:[],p:X}}function D(){X.r||a(X.c),X=X.p}function C(t,e){t&&t.i&&(L.delete(t),t.i(e))}function O(t,e,n,r){if(t&&t.o){if(L.has(t))return;L.add(t),X.c.push((()=>{L.delete(t),r&&(n&&t.d(1),r())})),t.o(e)}}function G(t,e,n){const r=t.$$.props[e];void 0!==r&&(t.$$.bound[r]=n,n(t.$$.ctx[r]))}function j(t){t&&t.c()}function q(t,e,n){const{fragment:l,on_mount:o,on_destroy:i,after_update:u}=t.$$;l&&l.m(e,n),B((()=>{const e=o.map(r).filter(s);i?i.push(...e):a(e),t.$$.on_mount=[]})),u.forEach(B)}function H(t,e){const n=t.$$;null!==n.fragment&&(a(n.on_destroy),n.fragment&&n.fragment.d(e),n.on_destroy=n.fragment=null,n.ctx=[])}function J(t,e){-1===t.$$.dirty[0]&&(E.push(t),z||(z=!0,T.then(P)),t.$$.dirty.fill(0)),t.$$.dirty[e/31|0]|=1<<e%31}function K(t,n,r,s,o,i,u=[-1]){const c=S;A(t);const d=n.props||{},g=t.$$={fragment:null,ctx:null,props:i,update:e,not_equal:o,bound:l(),on_mount:[],on_destroy:[],before_update:[],after_update:[],context:new Map(c?c.$$.context:[]),callbacks:l(),dirty:u,skip_bound:!1};let p=!1;if(g.ctx=r?r(t,d,((e,n,...r)=>{const l=r.length?r[0]:n;return g.ctx&&o(g.ctx[e],g.ctx[e]=l)&&(!g.skip_bound&&g.bound[e]&&g.bound[e](l),p&&J(t,e)),n})):[],g.update(),p=!0,a(g.before_update),g.fragment=!!s&&s(g.ctx),n.target){if(n.hydrate){const t=function(t){return Array.from(t.childNodes)}(n.target);g.fragment&&g.fragment.l(t),t.forEach(f)}else g.fragment&&g.fragment.c();n.intro&&C(t.$$.fragment),q(t,n.target,n.anchor),P()}A(c)}class U{$destroy(){H(this,1),this.$destroy=e}$on(t,e){const n=this.$$.callbacks[t]||(this.$$.callbacks[t]=[]);return n.push(e),()=>{const t=n.indexOf(e);-1!==t&&n.splice(t,1)}}$set(t){var e;this.$$set&&(e=t,0!==Object.keys(e).length)&&(this.$$.skip_bound=!0,this.$$set(t),this.$$.skip_bound=!1)}}function W(t){let n,r,l;return{c(){n=m("svg"),r=m("path"),x(r,"fill","currentColor"),x(r,"d",t[0]),x(n,"aria-hidden","true"),x(n,"class",l=u(t[1])+" svelte-1d15yci"),x(n,"role","img"),x(n,"xmlns","http://www.w3.org/2000/svg"),x(n,"viewBox",t[2])},m(t,e){d(t,n,e),c(n,r)},p(t,[e]){1&e&&x(r,"d",t[0]),2&e&&l!==(l=u(t[1])+" svelte-1d15yci")&&x(n,"class",l),4&e&&x(n,"viewBox",t[2])},i:e,o:e,d(t){t&&f(n)}}}function Z(t,e,r){let{icon:l}=e,a=[],s="",o="";return t.$$set=t=>{r(4,e=n(n({},e),i(t))),"icon"in t&&r(3,l=t.icon)},t.$$.update=()=>{8&t.$$.dirty&&r(2,o="0 0 "+l.icon[0]+" "+l.icon[1]),r(1,s="fa-svelte "+(e.class?e.class:"")),8&t.$$.dirty&&r(0,a=l.icon[4])},e=i(e),[a,s,o,l]}class tt extends U{constructor(t){super(),K(this,t,Z,W,o,{icon:3})}}
/*!
     * Font Awesome Free 5.15.1 by @fontawesome - https://fontawesome.com
     * License - https://fontawesome.com/license/free (Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT License)
     */var et={prefix:"far",iconName:"arrow-alt-circle-up",icon:[512,512,[],"f35b","M256 504c137 0 248-111 248-248S393 8 256 8 8 119 8 256s111 248 248 248zm0-448c110.5 0 200 89.5 200 200s-89.5 200-200 200S56 366.5 56 256 145.5 56 256 56zm20 328h-40c-6.6 0-12-5.4-12-12V256h-67c-10.7 0-16-12.9-8.5-20.5l99-99c4.7-4.7 12.3-4.7 17 0l99 99c7.6 7.6 2.2 20.5-8.5 20.5h-67v116c0 6.6-5.4 12-12 12z"]};function nt(t){let e,n,r=t[2].toFixed()+"";return{c(){e=p("span"),n=h(r),x(e,"class","svelte-6n71zn")},m(t,r){d(t,e,r),c(e,n)},p(t,e){4&e&&r!==(r=t[2].toFixed()+"")&&v(n,r)},d(t){t&&f(e)}}}function rt(t){let e,n,r=t[2].toFixed(1)+"";return{c(){e=p("span"),n=h(r),x(e,"class","svelte-6n71zn")},m(t,r){d(t,e,r),c(e,n)},p(t,e){4&e&&r!==(r=t[2].toFixed(1)+"")&&v(n,r)},d(t){t&&f(e)}}}function lt(t){let e,n,r;return n=new tt({props:{icon:et}}),{c(){e=p("div"),j(n.$$.fragment),w(e,"font-size","32px"),w(e,"transform","rotate("+t[4]+"deg)"),x(e,"class","svelte-6n71zn")},m(t,l){d(t,e,l),q(n,e,null),r=!0},p(t,n){(!r||16&n)&&w(e,"transform","rotate("+t[4]+"deg)")},i(t){r||(C(n.$$.fragment,t),r=!0)},o(t){O(n.$$.fragment,t),r=!1},d(t){t&&f(e),H(n)}}}function at(t){let n;return{c(){n=h("terminal state")},m(t,e){d(t,n,e)},p:e,i:e,o:e,d(t){t&&f(n)}}}function st(t){let n;return{c(){n=h("blocked")},m(t,e){d(t,n,e)},p:e,i:e,o:e,d(t){t&&f(n)}}}function ot(t){let e,n,r,l,a,s,o,i,u,g,m,y,w,b,S,A,Q,E,R,V,N,T,z,B,_,M,I,P,Y,L,X=t[3][it.up].toFixed(3)+"",G=t[3][it.left].toFixed(3)+"",j=t[3][it.right].toFixed(3)+"",q=t[3][it.down].toFixed(3)+"";function H(t,e){return(null==u||4&e)&&(u=!!(Math.abs(t[2])<10)),u?rt:nt}let J=H(t,-1),K=J(t);const U=[st,at,lt],W=[];function Z(t,e){return t[0]?0:t[1]?1:2}return A=Z(t),Q=W[A]=U[A](t),{c(){e=p("div"),n=p("div"),r=$(),l=p("div"),a=p("span"),s=h(X),o=$(),i=p("div"),K.c(),g=$(),m=p("div"),y=p("span"),w=h(G),b=$(),S=p("div"),Q.c(),E=$(),R=p("div"),V=p("span"),N=h(j),T=$(),z=p("div"),B=$(),_=p("div"),M=p("span"),I=h(q),P=$(),Y=p("div"),x(a,"class","svelte-6n71zn"),x(l,"class","sub-tile sub-tile-top svelte-6n71zn"),x(i,"class","sub-tile sub-tile-reward svelte-6n71zn"),k(i,"reward-is-negative",t[2]<0),x(y,"class","svelte-6n71zn"),x(m,"class","sub-tile sub-tile-left svelte-6n71zn"),x(S,"class","sub-tile sub-tile-middle svelte-6n71zn"),x(V,"class","svelte-6n71zn"),x(R,"class","sub-tile sub-tile-right svelte-6n71zn"),x(M,"class","svelte-6n71zn"),x(_,"class","sub-tile sub-tile-bottom svelte-6n71zn"),x(e,"class","tile svelte-6n71zn"),x(e,"style",t[5]),k(e,"blocked",t[0]),k(e,"terminal",t[1])},m(t,u){d(t,e,u),c(e,n),c(e,r),c(e,l),c(l,a),c(a,s),c(e,o),c(e,i),K.m(i,null),c(e,g),c(e,m),c(m,y),c(y,w),c(e,b),c(e,S),W[A].m(S,null),c(e,E),c(e,R),c(R,V),c(V,N),c(e,T),c(e,z),c(e,B),c(e,_),c(_,M),c(M,I),c(e,P),c(e,Y),L=!0},p(t,[n]){(!L||8&n)&&X!==(X=t[3][it.up].toFixed(3)+"")&&v(s,X),J===(J=H(t,n))&&K?K.p(t,n):(K.d(1),K=J(t),K&&(K.c(),K.m(i,null))),4&n&&k(i,"reward-is-negative",t[2]<0),(!L||8&n)&&G!==(G=t[3][it.left].toFixed(3)+"")&&v(w,G);let r=A;A=Z(t),A===r?W[A].p(t,n):(F(),O(W[r],1,1,(()=>{W[r]=null})),D(),Q=W[A],Q?Q.p(t,n):(Q=W[A]=U[A](t),Q.c()),C(Q,1),Q.m(S,null)),(!L||8&n)&&j!==(j=t[3][it.right].toFixed(3)+"")&&v(N,j),(!L||8&n)&&q!==(q=t[3][it.down].toFixed(3)+"")&&v(I,q),(!L||32&n)&&x(e,"style",t[5]),1&n&&k(e,"blocked",t[0]),2&n&&k(e,"terminal",t[1])},i(t){L||(C(Q),L=!0)},o(t){O(Q),L=!1},d(t){t&&f(e),K.d(),W[A].d()}}}const it=Object.freeze({up:0,right:1,down:2,left:3});function ut(t,e,n){let r,l,a,s,o=!1,i=!1,u=0,c="";const d=()=>{l=Math.max(...r),a=r.indexOf(l),n(4,s=(90*a).toFixed())},f=()=>{n(3,r=Array(4).fill().map((()=>Math.random()))),d()};return f(),[o,i,u,r,s,c,()=>o,()=>{n(0,o=!0)},()=>i,()=>{n(1,i=!0)},()=>u,t=>{n(2,u=t)},()=>a,t=>i?u:r[t],()=>i?u:l,(t,e)=>{n(3,r[t]=e,r),d()},f,t=>{let e,r,l,a=.35+.6*t;o||(e=0+255*a,r=110+145*a,l=210+45*a,n(5,c="background: rgb("+e+","+r+","+l+", 1);"))}]}class ct extends U{constructor(t){super(),K(this,t,ut,ot,o,{isBlocked:6,setBlocked:7,isTerminal:8,setTerminal:9,getReward:10,setReward:11,getPolicy:12,getQValue:13,getMaxQValue:14,setQValue:15,initQValues:16,setHeat:17})}get isBlocked(){return this.$$.ctx[6]}get setBlocked(){return this.$$.ctx[7]}get isTerminal(){return this.$$.ctx[8]}get setTerminal(){return this.$$.ctx[9]}get getReward(){return this.$$.ctx[10]}get setReward(){return this.$$.ctx[11]}get getPolicy(){return this.$$.ctx[12]}get getQValue(){return this.$$.ctx[13]}get getMaxQValue(){return this.$$.ctx[14]}get setQValue(){return this.$$.ctx[15]}get initQValues(){return this.$$.ctx[16]}get setHeat(){return this.$$.ctx[17]}}function dt(t,e,n){const r=t.slice();return r[22]=e[n],r[23]=e,r[24]=n,r}function ft(t,e,n){const r=t.slice();return r[22]=e[n],r[25]=e,r[26]=n,r}function gt(t){let e,n,r=t[26],l=t[24];const a=()=>t[19](e,r,l),s=()=>t[19](null,r,l);return e=new ct({props:{}}),a(),{c(){j(e.$$.fragment)},m(t,r){q(e,t,r),n=!0},p(t,n){r===t[26]&&l===t[24]||(s(),r=t[26],l=t[24],a());e.$set({})},i(t){n||(C(e.$$.fragment,t),n=!0)},o(t){O(e.$$.fragment,t),n=!1},d(t){s(),H(e,t)}}}function pt(t){let e,n,r=Array(t[0]),l=[];for(let e=0;e<r.length;e+=1)l[e]=gt(ft(t,r,e));const a=t=>O(l[t],1,1,(()=>{l[t]=null}));return{c(){for(let t=0;t<l.length;t+=1)l[t].c();e=h("")},m(t,r){for(let e=0;e<l.length;e+=1)l[e].m(t,r);d(t,e,r),n=!0},p(t,n){if(5&n){let s;for(r=Array(t[0]),s=0;s<r.length;s+=1){const a=ft(t,r,s);l[s]?(l[s].p(a,n),C(l[s],1)):(l[s]=gt(a),l[s].c(),C(l[s],1),l[s].m(e.parentNode,e))}for(F(),s=r.length;s<l.length;s+=1)a(s);D()}},i(t){if(!n){for(let t=0;t<r.length;t+=1)C(l[t]);n=!0}},o(t){l=l.filter(Boolean);for(let t=0;t<l.length;t+=1)O(l[t]);n=!1},d(t){g(l,t),t&&f(e)}}}function mt(t){let e,n,r=Array(t[1]),l=[];for(let e=0;e<r.length;e+=1)l[e]=pt(dt(t,r,e));const a=t=>O(l[t],1,1,(()=>{l[t]=null}));return{c(){e=p("div");for(let t=0;t<l.length;t+=1)l[t].c();x(e,"class","maze svelte-csffwp"),w(e,"grid-template-columns","repeat("+t[0]+", 100px)")},m(t,r){d(t,e,r);for(let t=0;t<l.length;t+=1)l[t].m(e,null);n=!0},p(t,[s]){if(7&s){let n;for(r=Array(t[1]),n=0;n<r.length;n+=1){const a=dt(t,r,n);l[n]?(l[n].p(a,s),C(l[n],1)):(l[n]=pt(a),l[n].c(),C(l[n],1),l[n].m(e,null))}for(F(),n=r.length;n<l.length;n+=1)a(n);D()}(!n||1&s)&&w(e,"grid-template-columns","repeat("+t[0]+", 100px)")},i(t){if(!n){for(let t=0;t<r.length;t+=1)C(l[t]);n=!0}},o(t){l=l.filter(Boolean);for(let t=0;t<l.length;t+=1)O(l[t]);n=!1},d(t){t&&f(e),g(l,t)}}}function ht(t,e,n){let{numX:r}=e,{numY:l}=e,{blocked:a=[]}=e,{terminal:s=[]}=e,{rewards:o=[]}=e,{defaultReward:i=0}=e,u=Array.from({length:r},(()=>Array.from({length:l},(()=>null))));Q((()=>{a.forEach((t=>{u[t[0]][t[1]].setBlocked()})),s.forEach((t=>{u[t[0]][t[1]].setTerminal()}));for(let t=0;t<l;t++)for(let e=0;e<r;e++)u[e][t].setReward(i);o.forEach((t=>{u[t[0]][t[1]].setReward(t[2])})),g()}));const c=t=>Math.floor(Math.random()*Math.floor(t)),d=t=>u[t[0]][t[1]].isBlocked(),f=t=>u[t[0]][t[1]].isTerminal(),g=()=>{for(let t=0;t<l;t++)for(let e=0;e<r;e++)u[e][t].initQValues();$()},p=(t,e,n)=>{t[0]>=0&&t[0]<r&&t[1]>=0&&t[1]<l?(u[t[0]][t[1]].setQValue(e,n),$()):console.log("ERROR: Invalid setQValue coordinates [",t[0],":",t[1],"] !")},m=t=>u[t[0]][t[1]].getPolicy(),h=t=>t[0]>=0&&t[0]<r&&t[1]>=0&&t[1]<l?u[t[0]][t[1]].getReward():(console.log("ERROR: Invalid getReward coordinates [",t[0],":",t[1],"] !"),0),$=()=>{let t=1e6,e=-1e6;for(let n=0;n<l;n++)for(let l=0;l<r;l++){let r=u[l][n].getMaxQValue();t>r&&(t=r),e<r&&(e=r)}let n=e-t;for(let e=0;e<l;e++)for(let l=0;l<r;l++){let r=u[l][e].getMaxQValue();u[l][e].setHeat((r-t)/n)}};return t.$$set=t=>{"numX"in t&&n(0,r=t.numX),"numY"in t&&n(1,l=t.numY),"blocked"in t&&n(3,a=t.blocked),"terminal"in t&&n(4,s=t.terminal),"rewards"in t&&n(5,o=t.rewards),"defaultReward"in t&&n(6,i=t.defaultReward)},[r,l,u,a,s,o,i,c,d,f,g,p,(t,e)=>{for(let n=0;n<4;n++)p(t,n,e[n])},(t,e)=>u[t[0]][t[1]].getQValue(e),t=>u[t[0]][t[1]].getMaxQValue(),m,()=>{for(;;){let t=[c(r),c(l)];if(!f(t)&&!d(t))return t}},(t,e)=>Math.random()<e?c(4):m(t),(t,e)=>{let n=[...t];return e==it.down&&t[1]<l-1&&(n[1]+=1),e==it.right&&t[0]<r-1&&(n[0]+=1),e==it.up&&t[1]>0&&(n[1]-=1),e==it.left&&t[0]>0&&(n[0]-=1),d(n)&&(n=[...t]),[n,h(n)]},function(t,e,r){R[t?"unshift":"push"]((()=>{u[e][r]=t,n(2,u)}))}]}class $t extends U{constructor(t){super(),K(this,t,ht,mt,o,{numX:0,numY:1,blocked:3,terminal:4,rewards:5,defaultReward:6,getRandomInt:7,isBlocked:8,isTerminal:9,initQValues:10,setQValue:11,setQValues:12,getQValue:13,getMaxQValue:14,getPolicy:15,getRandomStartState:16,getEpsilonGreedyAction:17,step:18})}get getRandomInt(){return this.$$.ctx[7]}get isBlocked(){return this.$$.ctx[8]}get isTerminal(){return this.$$.ctx[9]}get initQValues(){return this.$$.ctx[10]}get setQValue(){return this.$$.ctx[11]}get setQValues(){return this.$$.ctx[12]}get getQValue(){return this.$$.ctx[13]}get getMaxQValue(){return this.$$.ctx[14]}get getPolicy(){return this.$$.ctx[15]}get getRandomStartState(){return this.$$.ctx[16]}get getEpsilonGreedyAction(){return this.$$.ctx[17]}get step(){return this.$$.ctx[18]}}function yt(t){let n;return{c(){n=p("div"),x(n,"class","plot svelte-1qi4jkb")},m(e,r){d(e,n,r),t[12](n)},p:e,i:e,o:e,d(e){e&&f(n),t[12](null)}}}function xt(t,e,n){let r,{title:l=""}=e,{xIsLog:a=!1}=e,{data:s=[]}=e,{yIsLog:o=!1}=e,{yTitle:i=""}=e,{hasSecondY:u=!1}=e,{dataSecond:c=[]}=e,{ySecondIsLog:d=!1}=e,{ySecondTitle:f=""}=e;const g=()=>{let t=s.length?-.02*s.length:-.02,e=s.length?1.02*s.length:1.02,n={title:l,showlegend:!1,xaxis:{type:a?"log":"linear",range:[t,e]},yaxis:{title:i,type:o?"log":"linear",range:[-1.1,1.1],titlefont:{color:"#08C"},tickfont:{color:"#08C"},autorange:!0},margin:{autoexpand:!1,t:50,l:40,b:30,r:20}};""!=i&&(n.margin.l=60);let g=[{x:[...Array(s.length).keys()],y:s,mode:"lines",line:{shape:"spline"},type:"scatter"}];if(u){n.yaxis2={title:f,type:d?"log":"linear",range:[-1.1,1.1],overlaying:"y",side:"right",titlefont:{color:"#E60"},tickfont:{color:"#E60"},autorange:!0},n.margin.r=""!=f?60:40;let t={x:[...Array(c.length).keys()],y:c,mode:"lines",line:{shape:"spline"},type:"scatter",yaxis:"y2"};g.push(t)}Plotly.react(r,g,n,{displaylogo:!1})};return Q((()=>{g()})),t.$$set=t=>{"title"in t&&n(3,l=t.title),"xIsLog"in t&&n(4,a=t.xIsLog),"data"in t&&n(1,s=t.data),"yIsLog"in t&&n(5,o=t.yIsLog),"yTitle"in t&&n(6,i=t.yTitle),"hasSecondY"in t&&n(7,u=t.hasSecondY),"dataSecond"in t&&n(2,c=t.dataSecond),"ySecondIsLog"in t&&n(8,d=t.ySecondIsLog),"ySecondTitle"in t&&n(9,f=t.ySecondTitle)},[r,s,c,l,a,o,i,u,d,f,g,()=>{n(1,s=[]),n(2,c=[]),g()},function(t){R[t?"unshift":"push"]((()=>{r=t,n(0,r)}))}]}class vt extends U{constructor(t){super(),K(this,t,xt,yt,o,{title:3,xIsLog:4,data:1,yIsLog:5,yTitle:6,hasSecondY:7,dataSecond:2,ySecondIsLog:8,ySecondTitle:9,updatePlot:10,clearPlot:11})}get updatePlot(){return this.$$.ctx[10]}get clearPlot(){return this.$$.ctx[11]}}function wt(t,e,n){let r=[],{maxData:l=1e3}=e;return t.$$set=t=>{"maxData"in t&&n(0,l=t.maxData)},[l,t=>{for(;r.length>=l;)r.shift();r.push(t)},t=>{let e=r.length;if(e<=t)return[...r];let n=new Array(t),l=new Array(e);for(;t--;){let a=Math.floor(Math.random()*e);n[t]=r[a in l?l[a]:a],l[a]=--e in l?l[e]:e}return n},()=>[...r],()=>{r=[]}]}class bt extends U{constructor(t){super(),K(this,t,wt,null,o,{maxData:0,add:1,getBatch:2,getAll:3,clear:4})}get add(){return this.$$.ctx[1]}get getBatch(){return this.$$.ctx[2]}get getAll(){return this.$$.ctx[3]}get clear(){return this.$$.ctx[4]}}function kt(t,e,n){let r,{duelingQNet:l=!1}=e;class a extends tf.layers.Layer{constructor(){super({})}getClassName(){return"DuelingLayer"}computeOutputShape(t){return t[0]}call(t,e){return tf.tidy((()=>{const e=t[0],n=t[1];return e.sub(e.mean(1).reshape([-1,1])).add(n)}))}}Q((()=>{s()}));const s=()=>{null!=r&&r.dispose(),l?i():o(),r.compile({optimizer:tf.train.adam(.005),loss:"meanSquaredError",metrics:["accuracy"]})},o=()=>{r=tf.sequential(),r.add(tf.layers.dense({inputShape:[2],units:10,useBias:!0,activation:"tanh"})),r.add(tf.layers.dense({units:10,useBias:!0,activation:"tanh"})),r.add(tf.layers.dense({units:10,useBias:!0,activation:"tanh"})),r.add(tf.layers.dense({units:4,useBias:!0,activation:"linear"}))},i=()=>{const t=tf.input({shape:[2]}),e=tf.layers.dense({units:10,useBias:!0,activation:"tanh"}).apply(t),n=tf.layers.dense({units:10,useBias:!0,activation:"tanh"}).apply(e),l=tf.layers.dense({units:10,useBias:!0,activation:"tanh"}).apply(n),s=tf.layers.dense({units:4,useBias:!0,activation:"linear"}).apply(l),o=tf.layers.dense({units:10,useBias:!0,activation:"tanh"}).apply(n),i=tf.layers.dense({units:1,useBias:!0,activation:"linear"}).apply(o),u=(new a).apply([s,i]);r=tf.model({inputs:t,outputs:u})};return t.$$set=t=>{"duelingQNet"in t&&n(0,l=t.duelingQNet)},[l,s,async(t,e)=>{const n=tf.tensor2d(t,[t.length,t[0].length],"float32"),l=tf.tensor2d(e,[e.length,e[0].length],"float32");await r.fit(n,l,{batchSize:t.length,epochs:10,shuffle:!0,callbacks:{}}),n.dispose(),l.dispose()},t=>{let e;return tf.tidy((()=>{const n=tf.tensor2d(t,[t.length,t[0].length],"float32");e=r.predict(n).arraySync()})),e},(t,e,n)=>{let r=[];return t.forEach(((t,l)=>{t-=(n[l]+e[l])/2,t/=(n[l]-e[l])/2,r.push(t)})),r}]}class St extends U{constructor(t){super(),K(this,t,kt,null,o,{duelingQNet:0,initModel:1,fit:2,predict:3,normalize:4})}get initModel(){return this.$$.ctx[1]}get fit(){return this.$$.ctx[2]}get predict(){return this.$$.ctx[3]}get normalize(){return this.$$.ctx[4]}}function At(t,e,n){const r=t.slice();return r[62]=e[n],r}function Qt(t){let n,r,l,a=t[62].name+"";return{c(){n=p("option"),r=h(a),n.__value=l=t[62].name,n.value=n.__value},m(t,e){d(t,n,e),c(n,r)},p:e,d(t){t&&f(n)}}}function Et(t){let e,n,r,l,s,o,i,u,m,k,S,A,Q,E,V,N,T,z,M,I,P,Y,L,X,F,D,J,K,U,W,Z,tt,et,nt,rt,lt,at,st;function ot(e){t[26].call(null,e)}function it(e){t[27].call(null,e)}let ut={yTitle:"reward per episode",ySecondTitle:"steps per episode",hasSecondY:!1};void 0!==t[9]&&(ut.data=t[9]),void 0!==t[8]&&(ut.dataSecond=t[8]),r=new vt({props:ut}),t[25](r),R.push((()=>G(r,"data",ot))),R.push((()=>G(r,"dataSecond",it)));let ct=t[18],dt=[];for(let e=0;e<ct.length;e+=1)dt[e]=Qt(At(t,ct,e));let ft={numX:t[4],numY:t[5],blocked:t[0],terminal:t[1],rewards:t[2],defaultReward:t[3]};U=new $t({props:ft}),t[31](U);let gt={maxData:t[17]};Z=new bt({props:gt}),t[32](Z),et=new bt({props:{maxData:Tt}}),t[33](et);let pt={duelingQNet:t[16]};return rt=new St({props:pt}),t[34](rt),{c(){e=p("div"),n=p("div"),j(r.$$.fragment),o=$(),i=p("div"),u=h("EPISODE : "),m=h(t[11]),k=$(),S=p("div"),A=p("select");for(let t=0;t<dt.length;t+=1)dt[t].c();Q=$(),E=p("div"),V=p("input"),N=h("\n      DQN"),T=$(),z=p("div"),M=p("input"),I=h("\n      Dueling"),P=$(),Y=p("button"),Y.textContent="init",L=$(),X=p("button"),X.textContent="halt",F=$(),D=p("button"),D.textContent="run",J=$(),K=p("div"),j(U.$$.fragment),W=$(),j(Z.$$.fragment),tt=$(),j(et.$$.fragment),nt=$(),j(rt.$$.fragment),x(n,"class","narrow-box svelte-1k2e5o4"),x(i,"class","box svelte-1k2e5o4"),x(A,"class","svelte-1k2e5o4"),void 0===t[10]&&B((()=>t[28].call(A))),x(V,"type","checkbox"),x(V,"class","svelte-1k2e5o4"),x(E,"class","flexrow svelte-1k2e5o4"),x(M,"type","checkbox"),x(M,"class","svelte-1k2e5o4"),x(z,"class","flexrow svelte-1k2e5o4"),x(Y,"class","svelte-1k2e5o4"),x(X,"class","svelte-1k2e5o4"),x(D,"class","svelte-1k2e5o4"),x(S,"class","box svelte-1k2e5o4"),x(K,"class","narrow-box svelte-1k2e5o4"),x(e,"class","container svelte-1k2e5o4"),w(e,"width",16+100*t[4]+4*(t[4]-1)+"px")},m(l,a){d(l,e,a),c(e,n),q(r,n,null),c(e,o),c(e,i),c(i,u),c(i,m),c(e,k),c(e,S),c(S,A);for(let t=0;t<dt.length;t+=1)dt[t].m(A,null);b(A,t[10]),c(S,Q),c(S,E),c(E,V),V.checked=t[15],c(E,N),c(S,T),c(S,z),c(z,M),M.checked=t[16],c(z,I),c(S,P),c(S,Y),c(S,L),c(S,X),c(S,F),c(S,D),c(e,J),c(e,K),q(U,K,null),d(l,W,a),q(Z,l,a),d(l,tt,a),q(et,l,a),d(l,nt,a),q(rt,l,a),lt=!0,at||(st=[y(A,"change",t[28]),y(A,"change",t[21]),y(V,"change",t[29]),y(V,"change",t[21]),y(M,"change",t[30]),y(M,"change",t[21]),y(Y,"click",t[21]),y(X,"click",t[20]),y(D,"click",t[19])],at=!0)},p(t,n){const a={};if(!l&&512&n[0]&&(l=!0,a.data=t[9],_((()=>l=!1))),!s&&256&n[0]&&(s=!0,a.dataSecond=t[8],_((()=>s=!1))),r.$set(a),(!lt||2048&n[0])&&v(m,t[11]),262144&n[0]){let e;for(ct=t[18],e=0;e<ct.length;e+=1){const r=At(t,ct,e);dt[e]?dt[e].p(r,n):(dt[e]=Qt(r),dt[e].c(),dt[e].m(A,null))}for(;e<dt.length;e+=1)dt[e].d(1);dt.length=ct.length}263168&n[0]&&b(A,t[10]),32768&n[0]&&(V.checked=t[15]),65536&n[0]&&(M.checked=t[16]);const o={};16&n[0]&&(o.numX=t[4]),32&n[0]&&(o.numY=t[5]),1&n[0]&&(o.blocked=t[0]),2&n[0]&&(o.terminal=t[1]),4&n[0]&&(o.rewards=t[2]),8&n[0]&&(o.defaultReward=t[3]),U.$set(o),(!lt||16&n[0])&&w(e,"width",16+100*t[4]+4*(t[4]-1)+"px");Z.$set({});et.$set({});const i={};65536&n[0]&&(i.duelingQNet=t[16]),rt.$set(i)},i(t){lt||(C(r.$$.fragment,t),C(U.$$.fragment,t),C(Z.$$.fragment,t),C(et.$$.fragment,t),C(rt.$$.fragment,t),lt=!0)},o(t){O(r.$$.fragment,t),O(U.$$.fragment,t),O(Z.$$.fragment,t),O(et.$$.fragment,t),O(rt.$$.fragment,t),lt=!1},d(n){n&&f(e),t[25](null),H(r),g(dt,n),t[31](null),H(U),n&&f(W),t[32](null),H(Z,n),n&&f(tt),t[33](null),H(et,n),n&&f(nt),t[34](null),H(rt,n),at=!1,a(st)}}}const Rt=.1,Vt=.2,Nt=.9,Tt=2e3;function zt(t,e,n){let{blocked:r=[]}=e,{terminal:l=Array([0,0])}=e,{rewards:a=Array([0,0,1])}=e,{defaultReward:s=0}=e,{startState:o}=e,{numEpisodes:i=1e3}=e,{planningSteps:u=10}=e,{numX:c=5}=e,{numY:d=5}=e;const f=c*d;let g,p,m,h,$,y,x,v,w=[],b=[],k=0,S=0,A=0,E=!1,V=!1,N=0;const T=t=>v.normalize(t,[0,0],[c-1,d-1]);Q((()=>{for(let t=0;t<d;t++)for(let e=0;e<c;e++){let n=[e,t],r=T(n);y.add({state:n,normState:r})}}));const z=(t,e)=>{let n=e(t);g.setQValue(t.state,t.a,n)},B=(t,e)=>{if(t.normState=T(t.state),x.add(t),N++,N<10)return;N=0;const n=x.getBatch(100);let r=[];n.forEach((t=>{r.push(t.normState)}));let l=v.predict(r);n.forEach(((t,n)=>{l[n][t.a]=e(t)})),v.fit(r,l),y.getAll().forEach((t=>{let e=v.predict([t.normState]);g.setQValues(t.state,e[0])}))},_=t=>{let e;return e=g.isTerminal(t.stateNext)?t.r:t.r+Nt*g.getMaxQValue(t.stateNext),E?e:.8*g.getQValue(t.state,t.a)+Vt*e},M=t=>{let e,n,r;g.isTerminal(t)?(w.push(A),b.push(k),p.updatePlot(),q()):$=setTimeout((()=>{n=g.getEpsilonGreedyAction(t,Rt),[e,r]=g.step(t,n),E?B({state:t,a:n,r:r,stateNext:e},_):z({state:t,a:n,r:r,stateNext:e},_),t=[...e],k+=r,A++,M(t)}),0)},I=t=>{let e;return e=g.isTerminal(t.stateNext)?t.r:t.r+Nt*g.getQValue(t.stateNext,t.aNext),E?e:.8*g.getQValue(t.state,t.a)+Vt*e},P=(t,e)=>{let n,r,l;g.isTerminal(t)?(w.push(A),b.push(k),p.updatePlot(),q()):$=setTimeout((()=>{[n,l]=g.step(t,e),r=g.getEpsilonGreedyAction(n,Rt),E?B({state:t,a:e,r:l,stateNext:n,aNext:r},I):z({state:t,a:e,r:l,stateNext:n,aNext:r},I),t=[...n],e=Number(r),k+=l,A++,P(t,e)}),0)},Y=t=>{let e;if(g.isTerminal(t.stateNext))e=t.r;else{let n,r=0;n=.025;for(let e=0;e<4;e++)r+=n*g.getQValue(t.stateNext,e);n=.9;let l=g.getPolicy(t.stateNext);r+=n*g.getQValue(t.stateNext,l),e=t.r+Nt*r}return E?e:.8*g.getQValue(t.state,t.a)+Vt*e},L=(t,e)=>{let n,r,l;g.isTerminal(t)?(w.push(A),b.push(k),p.updatePlot(),q()):$=setTimeout((()=>{[n,l]=g.step(t,e),r=g.getEpsilonGreedyAction(n,Rt),E?B({state:t,a:e,r:l,stateNext:n,aNext:r},Y):z({state:t,a:e,r:l,stateNext:n,aNext:r},Y),t=[...n],e=Number(r),k+=l,A++,L(t,e)}),0)};let X=Array.from({length:c},(()=>Array.from({length:d},(()=>Array.from({length:4},(()=>null)))))),F=[];const D=()=>{let t=g.getRandomInt(F.length);return F[t]},C=t=>{let e,n,r;g.isTerminal(t)?(w.push(A),b.push(k),p.updatePlot(),q()):$=setTimeout((()=>{n=g.getEpsilonGreedyAction(t,Rt),[e,r]=g.step(t,n),z({state:t,a:n,r:r,stateNext:e},_),((t,e,n,r)=>{let l=t[0],a=t[1],s=!1;for(let t=0;t<F.length;t++)if(F[t][0]==l&&F[t][1]==a&&F[t][2]==e){s=!0;break}s||F.push([t,e]),X[l][a][e]=[r,n]})(t,n,r,e),t=[...e];for(let t=0;t<u;t++){let t,e,n,r;[t,e]=D(),[r,n]=X[t[0]][t[1]][e],z({state:t,a:e,r:n,stateNext:r},_)}k+=r,A++,C(t)}),0)};let O;const G=t=>{let e,n,r;g.isTerminal(t)?((()=>{for(let t=0;t<d;t++)for(let e=0;e<c;e++)for(let n=0;n<4;n++){let r=!1,l=0,a=1;for(let s=0;s<O.length;s++)e==O[s][0][0]&&t==O[s][0][1]&&n==O[s][1]&&(r=!0),r&&(l+=a*O[s][2],a*=Nt);if(r){let r=.8*g.getQValue([e,t],n)+Vt*l;g.setQValue([e,t],n,r)}}})(),w.push(A),b.push(k),p.updatePlot(),q()):$=setTimeout((()=>{n=g.getEpsilonGreedyAction(t,Rt),[e,r]=g.step(t,n),O.push([t,n,r]),t=[...e],k+=r,A++,G(t)}),0)},j=[{name:"Q-Learning",func:()=>{let t;k=0,A=0,t=o||g.getRandomStartState(),M(t)}},{name:"SARSA",func:()=>{let t,e;k=0,A=0,t=o||g.getRandomStartState(),e=g.getEpsilonGreedyAction(t,Rt),P(t,e)}},{name:"Expected SARSA",func:()=>{let t,e;k=0,A=0,t=o||g.getRandomStartState(),e=g.getEpsilonGreedyAction(t,Rt),L(t,e)}},{name:"Dyna-Q",func:()=>{let t;k=0,A=0,t=o||g.getRandomStartState(),C(t)}},{name:"Monte Carlo",func:()=>{let t;k=0,A=0,O=[],t=g.getRandomStartState(),G(t)}}],q=()=>{h=setTimeout((()=>{S<i&&(n(11,S++,S),j.forEach((t=>{m==t.name&&t.func()})))}),0)};return t.$$set=t=>{"blocked"in t&&n(0,r=t.blocked),"terminal"in t&&n(1,l=t.terminal),"rewards"in t&&n(2,a=t.rewards),"defaultReward"in t&&n(3,s=t.defaultReward),"startState"in t&&n(22,o=t.startState),"numEpisodes"in t&&n(23,i=t.numEpisodes),"planningSteps"in t&&n(24,u=t.planningSteps),"numX"in t&&n(4,c=t.numX),"numY"in t&&n(5,d=t.numY)},[r,l,a,s,c,d,g,p,w,b,m,S,y,x,v,E,V,f,j,q,()=>{h&&clearTimeout(h),$&&clearTimeout($)},()=>{n(11,S=0),n(8,w=[]),n(9,b=[]),g.initQValues(),p.clearPlot(),v.initModel(),x.clear()},o,i,u,function(t){R[t?"unshift":"push"]((()=>{p=t,n(7,p)}))},function(t){b=t,n(9,b)},function(t){w=t,n(8,w)},function(){m=function(t){const e=t.querySelector(":checked")||t.options[0];return e&&e.__value}(this),n(10,m),n(18,j)},function(){E=this.checked,n(15,E)},function(){V=this.checked,n(16,V)},function(t){R[t?"unshift":"push"]((()=>{g=t,n(6,g)}))},function(t){R[t?"unshift":"push"]((()=>{y=t,n(12,y)}))},function(t){R[t?"unshift":"push"]((()=>{x=t,n(13,x)}))},function(t){R[t?"unshift":"push"]((()=>{v=t,n(14,v)}))}]}class Bt extends U{constructor(t){super(),K(this,t,zt,Et,o,{blocked:0,terminal:1,rewards:2,defaultReward:3,startState:22,numEpisodes:23,planningSteps:24,numX:4,numY:5},[-1,-1,-1])}}const _t=document.getElementById("maze-shell-0"),Mt=document.getElementById("maze-shell-1"),It=document.getElementById("maze-shell-2"),Pt=document.getElementById("maze-shell-3"),Yt=document.getElementById("maze-shell-4"),Lt=_t?new Bt({target:_t,props:{numX:12,numY:3,planningSteps:10,blocked:Array(),terminal:Array([1,2],[2,2],[3,2],[4,2],[5,2],[6,2],[7,2],[8,2],[9,2],[10,2],[11,2]),rewards:Array([1,2,-10],[2,2,-10],[3,2,-10],[4,2,-10],[5,2,-10],[6,2,-10],[7,2,-10],[8,2,-10],[9,2,-10],[10,2,-10],[11,2,0]),defaultReward:-.1,startState:[0,2]}}):null,Xt=Mt?new Bt({target:Mt,props:{numX:7,numY:3,planningSteps:10,blocked:Array(),terminal:Array([1,2],[2,2],[3,2],[4,2],[5,2],[6,2]),rewards:Array([1,2,-10],[2,2,-10],[3,2,-10],[4,2,-10],[5,2,-10],[6,2,0]),defaultReward:-.1,startState:[0,2]}}):null,Ft=It?new Bt({target:It,props:{numX:5,numY:5,blocked:Array(),terminal:Array([1,1]),rewards:Array([1,1,1]),defaultReward:-.1}}):null,Dt=Pt?new Bt({target:Pt,props:{numX:5,numY:5,blocked:Array(),terminal:Array([1,1]),rewards:Array([1,1,1],[3,3,1]),defaultReward:-.1}}):null,Ct=Yt?new Bt({target:Yt,props:{numX:6,numY:4,blocked:Array([0,0],[3,2],[4,1],[3,3]),terminal:Array([1,1]),rewards:Array([1,1,1]),defaultReward:-.1}}):null;return t.comp0=Lt,t.comp1=Xt,t.comp2=Ft,t.comp3=Dt,t.comp4=Ct,Object.defineProperty(t,"__esModule",{value:!0}),t}({});
//# sourceMappingURL=bundle.js.map
