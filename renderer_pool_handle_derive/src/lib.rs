use proc_macro::TokenStream;
use quote::quote;

#[proc_macro_derive(HandledPoolHandle)]
pub fn handled_pool_handle_derive(input: TokenStream) -> TokenStream {
  let ast: syn::DeriveInput = syn::parse(input).unwrap();

  let name = &ast.ident;
  
  let gen = quote! {
    impl HandledPoolHandle for #name {
      fn generation(&self) -> u32 {
        return self.generation;
      }
      
      fn index(&self) -> usize {
        return self.index as _;
      }
      
      fn make(index: u32, generation: u32) -> Self {
        return Self {
          index,
          generation
        };
      }
    }
  };

  return gen.into();
}