library(quantmod)
#list of FX rates
FX_list = c('DEXCHUS',"DEXJPUS")

for (code in FX_list){
  data <- getSymbols(Symbols=code, src='FRED', auto.assign = FALSE)
  write.zoo(data, file=paste0(code,'.csv'), index.name='DATETIME', sep=',', col.names=T)
  }
