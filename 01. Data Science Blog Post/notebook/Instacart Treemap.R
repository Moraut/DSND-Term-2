library(treemap)
library(dplyr)


#Set working directory
dir= "D:/GIT REPO/DSND-Term-2/01. Data Science Blog Post/data"
setwd(dir)

#####################################
#Load datasets
#####################################

#orders
orders = read.csv('../../../../orders.csv')
head(orders,5)

#products
products = read.csv('products.csv')
head(products,5)

#order_product_train
op_train = read.csv('order_products__train.csv')
head(op_train,5)

#order_product_prior
op_prior = read.csv('../../../../order_products__prior.csv')
head(op_prior,5)

#departments
departments = read.csv('departments.csv')
head(departments,5)

#aisles
aisles = read.csv('aisles.csv')
head(aisles,5)


##################################################
# Create Master Data table
##################################################
df = bind_rows(op_train, op_prior) %>% left_join(orders, by='order_id')
head(df)

df_master = df %>% left_join(products, by='product_id') %>% left_join(aisles, by='aisle_id')%>% left_join(departments, by='department_id')
head(df_master)              

##################################################
# Treemap Plots: Data Processing
##################################################


#Number of products by department and aisles sized by number of products
tmap_1 <- products %>% group_by(department_id, aisle_id) %>%
                    summarize(n=n()) %>% left_join(departments,by="department_id") %>%
                    left_join(aisles,by="aisle_id")

#Plot
png(file="../misc/treemap/Department and Aisle by Number of Products.png")

treemap(tmap_1,
        index=c("department","aisle"),
        vSize="n",vColor="department",
        palette="Set3",
        title="Department and Aisle by number of product",
        sortID="-n", 
        border.col="#FFFFFF",
        type="categorical", 
        fontsize.legend = 0,
        bg.labels = "#FFFFFF")
dev.off()

#Save the plot

#Number of products by department and aisles sized by number of products
tmap_2<-df_master %>% 
  group_by(product_id) %>% 
  summarize(count=n()) %>% 
  left_join(products,by="product_id") %>% 
  ungroup() %>% 
  group_by(department_id,aisle_id) %>% 
  summarize(sumcount = sum(count)) %>% 
  left_join(tmap_1, by = c("department_id", "aisle_id")) %>% 
  mutate(onesize = 1)


#Plot
png(file="../misc/treemap/Department and Aisle by Number of Orders.png")
treemap(tmap_2,
        index=c("department","aisle"),
        vSize="onesize",vColor="department",
        palette="Set3",
        title="Department and Aisle by number of orders ",
        sortID="-sumcount", 
        border.col="#FFFFFF",
        type="categorical", 
        fontsize.legend = 0,
        bg.labels = "#FFFFFF")

dev.off()
