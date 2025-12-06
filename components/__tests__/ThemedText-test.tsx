import * as React from 'react';
import { act } from 'react';
import renderer from 'react-test-renderer';

import { ThemedText } from '../ThemedText';

it(`renders correctly`, async () => {
  let tree: renderer.ReactTestRendererJSON | renderer.ReactTestRendererJSON[] | null = null;

  await act(async () => {
    tree = renderer.create(<ThemedText>Snapshot test!</ThemedText>).toJSON();
  });

  expect(tree).toMatchSnapshot();
});
