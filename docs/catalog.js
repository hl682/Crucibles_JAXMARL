const search=document.getElementById('search'),family=document.getElementById('family'),theme=document.getElementById('theme'),cards=[...document.querySelectorAll('.card[data-search]')];
function filter(){let n=0;for(const card of cards){const show=(!family.value||card.dataset.family===family.value)&&(!theme.value||card.dataset.themes.split(' ').includes(theme.value))&&card.dataset.search.toLowerCase().includes(search.value.toLowerCase());card.hidden=!show;if(show)n++}document.getElementById('count').textContent=document.documentElement.lang==='zh-CN'?`${n} 款游戏`:`${n} games`}
search.addEventListener('input',filter);family.addEventListener('change',filter);theme.addEventListener('change',filter);
filter();
