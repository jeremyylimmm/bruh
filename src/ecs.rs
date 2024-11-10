use std::collections::HashMap;
use std::any::*;

#[derive(Copy, Clone)]
pub struct Entity {
  pub word: u64
}

pub struct World {
  generation: Vec<u32>,
  free_list: Vec<u32>,
  arrays: HashMap<TypeId, Box<dyn Any>>
}

struct ComponentArray<T> {
  packed: Vec<Entity>,
  sparse: Vec<u32>,
  data: Vec<T>
}

impl Entity {
  fn unsafe_index(&self) -> usize {
    return (self.word & 0xffffffff) as usize;
  }

  fn generation(&self) -> u32 {
    return (self.word >> 32) as u32;
  }
}

impl PartialEq<Entity> for Entity {
  fn eq(&self, other: &Entity) -> bool {
    return self.word == other.word;
  }
}

impl World {
  pub fn new() -> Self {
    return Self {
      generation: Vec::new(),
      free_list: Vec::new(),
      arrays: HashMap::new()
    }
  }

  pub fn create_entity(&mut self) -> Entity {
    if self.free_list.is_empty() {
      self.free_list.push(self.generation.len() as u32);
      self.generation.push(1);
    }

    let index = self.free_list.pop().unwrap();
    let generation = self.generation[index as usize];

    return Entity {
      word: (generation as u64) << 32 | (index as u64)
    };
  }

  fn verify_entity(&self, entity: Entity) -> usize {
    let index = entity.unsafe_index();

    assert!(index < self.generation.len());
    assert!(self.generation[index] == entity.generation());

    return index as usize;
  }

  pub fn destroy_entity(&mut self, entity: Entity) {
    let index = self.verify_entity(entity);
    self.generation[index] += 1;
    self.free_list.push(index as u32);
  }

  fn get_array<T: 'static>(&mut self) -> &mut ComponentArray<T> {
    if !self.arrays.contains_key(&TypeId::of::<T>()) {
      self.arrays.insert(TypeId::of::<T>(), Box::new(ComponentArray::<T>::new()));
    }

    return self.arrays.get_mut(&TypeId::of::<T>()).unwrap().downcast_mut::<ComponentArray<T>>().unwrap();
  }

  pub fn add<T: 'static>(&mut self, entity: Entity, item: T) {
    self.verify_entity(entity);
    return self.get_array().add(entity, item); 
  }

  pub fn get<T: 'static>(&mut self, entity: Entity) -> &mut T {
    self.verify_entity(entity);
    return self.get_array().get(entity);
  }

  pub fn remove<T: 'static>(&mut self, entity: Entity) {
    self.verify_entity(entity);
    return self.get_array::<T>().remove(entity);
  }

  pub fn view<'a, T: 'static>(&'a mut self) -> ComponentArrayIterator<'a, T> {
    return self.get_array::<T>().iterate();
  }
}

impl<T> ComponentArray<T> {
  fn new() -> Self {
    return Self {
      packed: Vec::new(),
      sparse: Vec::new(),
      data: Vec::new()
    };
  }

  fn add(&mut self, entity: Entity, item: T) {
    while (entity.unsafe_index() as usize) >= self.sparse.len() {
      self.sparse.push(0xffffffff);
    }

    assert!(self.sparse[entity.unsafe_index()] == 0xffffffff);
    let index = self.packed.len();
    
    self.packed.push(entity);
    self.data.push(item);

    self.sparse[entity.unsafe_index()] = index as u32;
  }

  fn assert_exists(&self, entity: Entity) {
    assert!(entity.unsafe_index() < self.sparse.len());
    assert!(self.sparse[entity.unsafe_index()] != 0xffffffff);
  }

  fn get(&mut self, entity: Entity) -> &mut T {
    self.assert_exists(entity);
    return &mut self.data[self.sparse[entity.unsafe_index()] as usize];
  }

  fn remove(&mut self, entity: Entity) {
    self.assert_exists(entity);

    let index = self.sparse[entity.unsafe_index()] as usize;

    let last_entity = self.packed.pop().unwrap();
    let last_item = self.data.pop().unwrap();

    if last_entity != entity {
      self.data[index] = last_item;
      self.packed[index] = last_entity;
      self.sparse[last_entity.unsafe_index()] = index as u32;
    }

    self.sparse[entity.unsafe_index()] = 0xffffffff;
  }

  fn iterate<'a>(&'a mut self) -> ComponentArrayIterator<'a, T> {
    return ComponentArrayIterator::<'a, T> {
      item: self.data.iter_mut(),
      entity: self.packed.iter()
    };
  }
}

pub struct ComponentArrayIterator<'a, T> {
  item: std::slice::IterMut<'a, T>,
  entity: std::slice::Iter<'a, Entity>
}

impl<'a, T> Iterator for ComponentArrayIterator<'a, T> {
  type Item = (&'a mut T, Entity);

  fn next(&mut self) -> Option<Self::Item> {
    if let Some(item) = self.item.next() {
      let entity = self.entity.next().unwrap();
      return Some((item, *entity));
    }

    return None;
  }
}